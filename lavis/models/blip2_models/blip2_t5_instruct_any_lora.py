"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import string
import random
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

import transformers
from transformers.activations import ACT2FN
from peft import LoraConfig, get_peft_model

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import lavis.models.blip2_models.Qformer_lora as Qformer_lora 
from lavis.common.utils import is_url
from lavis.common.dist_utils import download_cached_file
from lavis.models.blip2_models.Qformer_lora import lora, custom_lora, mark_only_lora_as_trainable, check_lora_application


@registry.register_model("blip2_t5_instruct_any_lora")
class Blip2T5InstructAnyLoRA(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - flant5xl
        - flant5xxl
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5_instruct", "flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xl": "configs/models/blip2/blip2_instruct_flant5xl_qformer_lora.yaml",
        "flant5xxl": "configs/models/blip2/blip2_instruct_flant5xxl_qformer_lora.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        num_few_shot_examples=0,
        few_shot_prob=0,
        qformer_text_input=True,
        qformer_video_input=False,
        qformer_max_video_frame_count=8,
        qformer_use_lora=True,
        qformer_num_classes=None,
        llm_lora_r=8,
        llm_lora_apply="attn",
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        # ------------------------------------ ViT ----------------------------------- #

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # --------------------------------- Q-Former --------------------------------- #

        self.qformer_video_input = qformer_video_input
        frame_width = qformer_max_video_frame_count if qformer_video_input else 1

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features * frame_width
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        
        # Train only the Qformer LoRA
        if qformer_use_lora:
            mark_only_lora_as_trainable(self.Qformer)
            check_lora_application(self.Qformer)

        num_params = sum([p.numel() for p in self.Qformer.parameters() if p.requires_grad])
        print(f"Number of trainable parameters in Qformer: {num_params}")

        # --------------------------------- LLM Head --------------------------------- #
        
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='right')

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False 
            param.data = param.data.bfloat16()
        
        if llm_lora_apply:
            self._convert_t5_lora(llm_lora_r, llm_lora_apply)
        
        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        # Regression Head

        self.fixed_cls = None
        if qformer_num_classes:
            self.fixed_cls = QFormerClassHead(self.Qformer.config.hidden_size, qformer_num_classes)

        # -------------------------------- Parameters -------------------------------- #

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_prob = few_shot_prob

        self.qformer_text_input = qformer_text_input

    def _convert_t5_lora(self, llm_lora_r=8, llm_lora_apply="attn"):
        # def _find_all_linear_names(model):
        #     cls = torch.nn.Linear
        #     lora_module_names = set()
        #     module_names = set()
        #     for name, module in model.named_modules():
        #         print(f"all print :{type(module)}")
        #         module_names.add(name)
        #         if isinstance(module, cls):
        #             print(name)
        #             names = name.split('.')
        #             lora_module_names.add('.'+names[0] if len(names) == 1 else '.'+names[-1])
        #     print(f"1st val {list(module_names)}")
        #     print(f"2nd val {list(lora_module_names)}")
        #     # if 'lm_head' in lora_module_names: # needed for 16-bit
        #     #     lora_module_names.remove('lm_head')
        #     return list(lora_module_names)
        
        target_modules = []
        if llm_lora_apply == "attn":
            target_modules = ['q','v'] 
        elif llm_lora_apply == "ffn":
            target_modules = ["wi", "wo", "wi_1", "wi_0"]
        elif llm_lora_apply == "all":
            target_modules = ['q', 'v', "wi", "wo", "wi_1", "wi_0"] 
        else: 
            print("Wrong llm_lora_apply value in yaml!!")
        print(f"applying llm lora on {llm_lora_apply}")
        lora_config = LoraConfig(
            r=llm_lora_r,
            lora_alpha=8,
            target_modules=target_modules, #_find_all_linear_names(self.t5_model),
            # lora_dropout=training_args.lora_dropout,
            # bias=training_args.lora_bias,
            task_type="SEQ_2_SEQ_LM",
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        self.t5_model.print_trainable_parameters()
        

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        image = samples["image"]
        prompt = samples["text_input"]

        query_tokens, query_atts = self._get_query_tokens(image)
        text_tokens = self._get_qformer_text_input(prompt, image)
        inputs_t5, atts_t5, q_output = self._encode_qformer_t5(image, query_tokens, query_atts, text_tokens, return_query_output=True)
        cls_output = self._get_class_output(q_output, query_tokens, targets=samples['targets']) if samples.get('targets') is not None else None
        
        cls_loss = 0
        if cls_output is not None:
            cls_loss = cls_output['loss']

        fs_embeds, fs_atts = None, None
        if self.few_shot_prob > 0 and "few_shot_samples" in samples:
            fs_embeds, fs_atts = self.prepare_few_shot_embeds(samples['few_shot_samples'])

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_output_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(image.device)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            inputs_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            if fs_embeds is not None:
                inputs_embeds = torch.cat([fs_embeds, inputs_embeds], dim=1)
                inputs_atts = torch.cat([fs_atts, inputs_atts], dim=1)

            # input_embeds = [few shot, images, text query]
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            lm_loss = outputs.loss

            return {"loss": lm_loss + cls_loss, "lm_loss": lm_loss, "cls_loss": cls_loss}

    def prepare_few_shot_embeds(self, samples):
        this_n_fs = random.choices(
            list(range(self.num_few_shot_examples + 1)),
            weights=[1 - self.few_shot_prob] + [self.few_shot_prob / self.num_few_shot_examples] * self.num_few_shot_examples
        )[0]

        if this_n_fs == 0:
            return None, None

        images = []
        text_input = []
        for sample in samples:
            for n in range(this_n_fs):
                images.append(sample['image'][n])
                text_input.append(sample['text_input'][n])
        images = torch.stack(images, dim=0)

        query_tokens, query_atts = self._get_query_tokens(images)
        text_tokens = self._get_qformer_text_input(text_input, images)
        inputs_t5, atts_t5 = self._encode_qformer_t5(images, query_tokens, query_atts, text_tokens)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(images.device)

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            inputs_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        if this_n_fs > 1:
            inputs_embeds = inputs_embeds.reshape(inputs_embeds.size(0) // this_n_fs, inputs_embeds.size(1) * this_n_fs, inputs_embeds.size(2))
            inputs_atts = inputs_atts.reshape(inputs_atts.size(0) // this_n_fs, inputs_atts.size(1) * this_n_fs)

        return inputs_embeds, inputs_atts

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        prompt = samples.get('prompt', self.prompt)
        image = samples["image"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        query_tokens, query_atts = self._get_query_tokens(image)
        text_tokens = self._get_qformer_text_input(prompt, image)

        inputs_t5, atts_t5, q_output = self._encode_qformer_t5(image, query_tokens, query_atts, text_tokens, return_query_output=True)

        input_tokens = self.t5_tokenizer(prompt, padding="longest", return_tensors="pt").to(image.device)
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    @torch.no_grad()
    def generate_qformer(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if use_nucleus_sampling:
            num_beams = 1
        image = samples["image"]

        # encode image
        image_embeds = self.ln_vision(self.visual_encoder(image))
        if num_beams > 1:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        # get query
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # fake text input
        input_ids = torch.LongTensor(image.size(0), 1).fill_(self.tokenizer.bos_token_id).to(image.device)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                        for i in range(len(samples["text_input"]))
                    ]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if self._apply_lemmatizer or samples.get("apply_lemmatizer", True):
            output_text = self._lemmatize(output_text)

        return output_text


    def class_head(self, samples):
        image = samples["image"]
        prompt = samples["text_input"]

        query_tokens, query_atts = self._get_query_tokens(image)
        text_tokens = self._get_qformer_text_input(prompt, image)
        q_output = self._encode_qformer(image, query_tokens, query_atts, text_tokens)
        cls_output = self._get_class_output(q_output, query_tokens)
        return cls_output['prediction']


    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if isinstance(candidates[0], list):
            results = []
            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0)
                }

                if "text_input" in samples:
                    this_sample["text_input"] = [samples["text_input"][i]]
                # if 'context' in samples:
                #     this_sample['context'] = [samples["context"][i]]
                # if 'history' in samples:
                #     this_sample['history'] = [samples["history"][i]]
                # if 'caption' in samples:
                #     this_sample['caption'] = [samples["caption"][i]]
                results.append(self._predict_class(this_sample, candidates[i], n_segments))

            try:
                results = torch.cat(results, dim=0)
            except:
                # results = [[a], [b], [c]] -> [a, b, c]
                results = [res[0] for res in results]
            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - prompt: the instruction
            candidates:
                (list): A list of candidate class names;
            n_segments:
                (int): Split the candidates into n_segments and predict one by one. This is useful when the number of candidates is too large.
        Returns:
            output_class: predicted class index
        """

        image = samples["image"]
        # prompt = samples["prompt"]
        prompt = [samples['text_input'][i] for i in range(len(image))]
        bs = image.size(0)
        n_cands = len(candidates)


        # Encode Q-Former
        query_tokens, query_atts = self._get_query_tokens(image)
        text_tokens = self._get_qformer_text_input(prompt, image)
        inputs_t5, atts_t5 = self._encode_qformer_t5(image, query_tokens, query_atts, text_tokens)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(prompt, padding="longest", return_tensors="pt").to(image.device)
            output_tokens = self.t5_tokenizer(candidates, padding="longest", return_tensors="pt").to(image.device)
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            inputs_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            encoder_outputs = self.t5_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs_atts,
            )

            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                # this_encoder_outputs = copy.deepcopy(encoder_outputs)
                this_encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0].clone(),
                )

                this_encoder_outputs['last_hidden_state'] = this_encoder_outputs[0].repeat_interleave(seg_len, dim=0)
                this_encoder_atts = inputs_atts.repeat_interleave(seg_len, dim=0)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len
                this_output_tokens_ids = output_tokens.input_ids[start_i:end_i].repeat(bs, 1)
                this_output_tokens_atts = output_tokens.attention_mask[start_i:end_i].repeat(bs, 1)

                this_targets = this_output_tokens_ids.masked_fill(this_output_tokens_ids == self.t5_tokenizer.pad_token_id, -100)

                outputs = self.t5_model(
                    encoder_outputs=this_encoder_outputs,
                    attention_mask=this_encoder_atts,
                    decoder_attention_mask=this_output_tokens_atts,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )
                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)
            # top_predicted_classes = [candidates[idx] for idx in output_class_ranks[:, 0].tolist()]

            # encoder_outputs['last_hidden_state'] = encoder_outputs[0].repeat_interleave(n_cands, dim=0)
            # encoder_atts = encoder_atts.repeat_interleave(n_cands, dim=0)
            # output_tokens.input_ids = output_tokens.input_ids.repeat(bs, 1)
            # output_tokens.attention_mask = output_tokens.attention_mask.repeat(bs, 1)

            # # compute the LM loss for each candidate (sum logprob across all tokens) and select the highest
            # targets = output_tokens.input_ids.masked_fill(output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100)

            # outputs = self.t5_model(
            #     encoder_outputs=encoder_outputs,
            #     attention_mask=encoder_atts,
            #     decoder_attention_mask=output_tokens.attention_mask,
            #     return_dict=True,
            #     labels=targets,
            #     reduction="none",
            # )
            # loss = outputs.loss

            # loss = loss.reshape(bs, n_cands)
            # output_class_ranks = torch.argsort(loss, dim=-1) # (bs, num_candidates)

        # return top_predicted_classes
        return output_class_ranks


    def match(self, samples):
        image = samples["image"]
        prompt = samples["text_input"]
        query_tokens, query_atts = self._get_query_tokens(image)
        text_output = self._get_qformer_text_input(prompt, image)

        query_output = self._encode_qformer(image, query_tokens, query_atts, text_output)
        itm_embeddings = query_output.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(itm_embeddings)
        itm_logit = itm_logit.mean(dim=1)
        return itm_logit


    def contrast(self, samples):
        image = samples["image"]
        prompt = samples["text_input"]

        query_tokens, query_atts = self._get_query_tokens(image)
        text_input = self._get_qformer_text_input(prompt, image)

        query_output = self._encode_qformer(image, query_tokens, query_atts)
        text_output = self.Qformer.bert(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            return_dict=True,
        )

        image_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        sims = torch.bmm(image_feats, text_feat.unsqueeze(-1))
        sim, _ = torch.max(sims, dim=1)
        return sim


    def caption(self, samples):
        image = samples["image"]
        text = samples["text_input"]

        # encode image and query
        image_embeds, image_atts = self._encode_vision(image)
        query_tokens, query_atts = self._get_query_tokens(image)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        # tokenize text
        text_tokens = self._get_qformer_text_input(text, image)
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        # decode text from encoded image
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=torch.cat([query_atts, text_tokens.attention_mask], dim=1),
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )
        return lm_output



    # ---------------------------------------------------------------------------- #
    #                                 Core Q-Former                                #
    # ---------------------------------------------------------------------------- #

    def _get_query_tokens(self, image):
        query_tokens = self.query_tokens #.expand(image.size(0), -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(query_tokens.device)
        return query_tokens, query_atts
    
    def _get_qformer_text_input(self, prompt, image):
        text_output = self.tokenizer(
            prompt,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt"
        ).to(image.device)
        return text_output

    def _encode_vision(self, image):
        # TODO possible to encode video here? if image.dim() == 5:
        with self.maybe_autocast():
            if image.dim() == 5:
                B, T, C, H, W = image.size()
                image = image.reshape(B*T, C, H, W)
                image_embeds = self.ln_vision(self.visual_encoder(image))
                _, L, C = image_embeds.size()
                image_embeds = image_embeds.reshape(B, T*L, C)
            else:
                image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        return image_embeds, image_atts
    
    def _encode_qformer(self, image, query_tokens, query_atts, text_tokens=None):
        # add batch size
        query_tokens = query_tokens.expand(image.size(0), -1, -1)
        if query_atts is not None:
            query_atts = query_atts.expand(image.size(0), -1)

        # get image features
        image_embeds, image_atts = self._encode_vision(image)
        
        # encode q former
        if self.qformer_text_input and text_tokens is not None:
            query_output = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=torch.cat([query_atts, text_tokens.attention_mask], dim=1),
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        return query_output


    def _encode_qformer_t5(self, image, query_tokens, Qformer_atts=None, text_tokens=None, return_query_output=False):
        if image.dim() == 5 and not self.qformer_video_input:
            inputs_t5, atts_t5, query_outputs = [], [], []
            for j in range(image.size(1)):
                frame_inputs_t5, frame_atts_t5, query_output = self._encode_qformer_t5_single_image(image[:,j,:,:,:], query_tokens, Qformer_atts, text_tokens)
                inputs_t5.append(frame_inputs_t5)
                atts_t5.append(frame_atts_t5)
                query_outputs.append(query_output)
            inputs_t5 = torch.cat(inputs_t5, dim=1)
            atts_t5 = torch.cat(atts_t5, dim=1)
        else:
            frame_inputs_t5, frame_atts_t5, query_output = self._encode_qformer_t5_single_image(image, query_tokens, Qformer_atts, text_tokens)
            # query_output = [query_output]
        
        if return_query_output:
            return frame_inputs_t5, frame_atts_t5, query_output
        return frame_inputs_t5, frame_atts_t5

    def _encode_qformer_t5_single_image(self, image, query_tokens, query_atts, text_tokens):
        query_output = self._encode_qformer(image, query_tokens, query_atts, text_tokens)
        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_t5, atts_t5, query_output

    def _get_class_output(self, q_output, query_tokens, targets=None):
        if self.fixed_cls is not None:
            x = q_output.last_hidden_state
            x = x[:,:query_tokens.size(1),:]
            return self.fixed_cls(x, targets=targets)

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        num_few_shot_examples = cfg.get("num_few_shot_examples", 0)
        few_shot_prob = cfg.get("few_shot_prob", 0.0)

        qformer_text_input = cfg.get("qformer_text_input", True)
        qformer_video_input = cfg.get("qformer_video_input", False)
        qformer_max_video_frame_count = cfg.get("qformer_max_video_frame_count", 8)

        qformer_num_classes = cfg.get("qformer_num_classes", None)
        
        # TODO: if you want to control PEFT by config, you should add some varaibles here
        llm_lora_r = cfg.get("llm_lora_r", 8)
        llm_lora_apply = cfg.get("llm_lora_apply", False) 
        r = cfg.get("lora_r", 8)
        alpha = cfg.get("lora_alpha", 16)
        dropout = cfg.get("lora_dropout", 0.05)
        
        self_attention_qv_lora = cfg.get("self_attention_qv_lora", False)
        self_attention_output_lora = cfg.get("self_attention_output_lora", False)
        ffn_lora = cfg.get("ffn_lora", False)
        
        qformer_crossattention_lora_q = cfg.get("qformer_crossattention_lora_q", False)
        qformer_crossattention_lora_k = cfg.get("qformer_crossattention_lora_k", False)
        qformer_crossattention_lora_v = cfg.get("qformer_crossattention_lora_v", False)
        qformer_crossattention_lora_o = cfg.get("qformer_crossattention_lora_o", False)
        qkv = [qformer_crossattention_lora_q, qformer_crossattention_lora_k, qformer_crossattention_lora_v]

        with lora(r, alpha, dropout, enabled=self_attention_qv_lora, qkv=qkv), \
             custom_lora(r, alpha, dropout, enabled=(self_attention_output_lora or qformer_crossattention_lora_o), type="BertSelfOutput", sc=[self_attention_output_lora, qformer_crossattention_lora_o]), \
             custom_lora(r, alpha, dropout, enabled=ffn_lora, type="BertOutput"):
            model = cls(
                vit_model=vit_model,
                img_size=img_size,
                drop_path_rate=drop_path_rate,
                use_grad_checkpoint=use_grad_checkpoint,
                vit_precision=vit_precision,
                freeze_vit=freeze_vit,
                num_query_token=num_query_token,
                t5_model=t5_model,
                prompt=prompt,
                max_txt_len=max_txt_len,
                max_output_txt_len=max_output_txt_len,
                apply_lemmatizer=apply_lemmatizer,
                num_few_shot_examples=num_few_shot_examples,
                few_shot_prob=few_shot_prob,
                qformer_text_input=qformer_text_input,
                qformer_video_input=qformer_video_input,
                qformer_max_video_frame_count=qformer_max_video_frame_count,
                qformer_num_classes=qformer_num_classes,
                qformer_use_lora=self_attention_qv_lora or any(qkv) or self_attention_output_lora or qformer_crossattention_lora_o,
                llm_lora_r=llm_lora_r,
                llm_lora_apply=llm_lora_apply
            )

        model.load_checkpoint_from_config(cfg)

        return model

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Qformer.bert.encoder.layer.0.crossattention.self.key.weight

        # strict=False for peft layers
        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg



# class QFormerClassHead(nn.Module):
#     def __init__(self, input_size, n_classes, hidden_act='gelu'):
#         super().__init__()
#         self.fn1 = nn.Linear(input_size, input_size)
#         self.gru = nn.GRU(input_size, input_size, bidirectional=True, batch_first=True)
#         self.fn2 = nn.Linear(input_size * 2, n_classes)
#         self.act = ACT2FN[hidden_act]
#         # self.loss = nn.BCELoss(reduction='none')
#         self.loss = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, tokens, hidden_state=None, targets=None):
#         x = self.fn1(tokens)
#         x = self.act(x)
#         x, hidden_state = self.gru(x, hidden_state)
#         x = x[:, 0]
#         x = self.act(x)
#         logits = x = self.fn2(x)
#         loss = 0
#         if targets is not None:
#             loss = self.loss(logits, targets.float())
#             loss = (loss[targets >= 0]).sum() / max(1, (targets >= 0).sum())
#         y = F.sigmoid(x)
#         return {
#             'loss': loss,
#             'hidden_state': hidden_state,
#             'prediction': y,
#         }
    
class QFormerClassHead(nn.Module):
    def __init__(self, input_size, n_classes, nhead=8, num_encoder_layers=1, seq_len=500, hidden_act='gelu'):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_size))
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len + 1, input_size))  # Assuming max seq length + cls token
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, activation=hidden_act)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)
        self.fn1 = nn.Linear(input_size, input_size)
        self.fn2 = nn.Linear(input_size, n_classes)
        self.act = nn.GELU()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, tokens, targets=None):
        cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # Add positional encodings
        tokens += self.pos_encoder[:, :tokens.size(1), :]

        x = self.fn1(tokens)
        x = self.act(x)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.act(x)
        logits = self.fn2(x)

        loss = 0
        if targets is not None:
            loss = self.loss(logits, targets.float())
            loss = (loss[targets >= 0]).sum() / max(1, (targets >= 0).sum())
        
        y = torch.sigmoid(logits)
        return {
            'loss': loss,
            'prediction': y,
        }