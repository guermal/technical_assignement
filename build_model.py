import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig,AutoTokenizer,AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
import inspect

class VideoClassificationModel(nn.Module):
    def __init__(self, model_name, num_classes, freeze_backbone=False):
        super().__init__()

        # Load backbone encoder only (no classification head)
        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)

        # Get hidden size from config
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'hidden_dim', None))
        assert hidden_size is not None, "Backbone config must define hidden_size or hidden_dim"

        # Custom classifier head
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        # Optionally freeze backbone (can be overridden later if using LoRA)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        if isinstance(self.backbone, PeftModel):
            
            outputs = self.backbone.base_model(pixel_values=inputs)
        else:
            outputs = self.backbone(inputs)
        
        outputs = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(outputs),outputs


class VisionLanguageModel(nn.Module):
    def __init__(self,vision_model,llm_model,num_classes,vision_target_modules,llm_target_modules,captioning=False,use_text=False,use_lora_vison=False,freeze_visionbackbone=False,freeze_llmbackbone=False):
        super().__init__()
        self.vision_backbone,vision_hidden_size = build_vision_model(vision_model,vision_target_modules,num_classes,use_lora_vison,freeze_visionbackbone)
        self.use_text = use_text
        #self.vision_backbone = self.vision_backbone.backbone
        self.captioning = captioning
        llm_config = AutoConfig.from_pretrained(llm_model)
        if self.captioning:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm_backbone = AutoModelForCausalLM.from_pretrained(llm_model)

        else:
            
            
            
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = "[PAD]"
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            
            
            self.llm_backbone = AutoModel.from_pretrained(llm_model, config=llm_config)
            
        # Get hidden size from config
        hidden_size = getattr(llm_config, 'hidden_size', getattr(llm_config, 'hidden_dim', None))
        assert hidden_size is not None, "Backbone config must define hidden_size or hidden_dim"
        if hidden_size != vision_hidden_size:
            self.fc1 = nn.Linear(vision_hidden_size,hidden_size)
        else:
            self.fc1 = None
        # Custom classifier head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Optionally freeze backbone (can be overridden later if using LoRA)
        if freeze_llmbackbone:
            for param in self.llm_backbone.parameters():
                param.requires_grad = False
        
    def forward(self,inputs,text_inputs=None):
        
        #_,features = self.vision_backbone(inputs).last_hidden_state # B,T,D
        _,features = self.vision_backbone(inputs)
        #features = features.mean(1).unsqueeze(1)
        features = features.unsqueeze(1)
        if self.fc1 is not None:
            features = self.fc1(features)
        B = features.size(0)
        if self.captioning:
            if text_inputs is not None:
                # Training mode: prepare input_ids and labels
                tokenized = self.tokenizer(
                    text_inputs,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt"
                ).to(inputs.device)
                input_ids = tokenized["input_ids"]

                # Get token embeddings
                text_embeds = self.llm_backbone.get_input_embeddings()(input_ids)
                ignore_index = -100
                B, _, _ = features.shape
                pad = torch.full((B, 1), ignore_index, dtype=input_ids.dtype, device=input_ids.device)
                padded_labels = torch.cat([pad,input_ids], dim=1)  # [B, 9]
                
                # Concatenate vision features and text input embeddings
                inputs_embeds = torch.cat([features,text_embeds], dim=1)
                #print(inputs_embeds.size(),text_embeds.size(),features.size())
                outputs = self.llm_backbone(inputs_embeds=inputs_embeds, labels=padded_labels)
                pred_ids = torch.argmax(outputs.logits, dim=-1)  # [B, SeqLen]
                decoded_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

                return self.llm_backbone(inputs_embeds=inputs_embeds, labels=padded_labels),decoded_texts
            else:
                text_inputs = ['a']
                tokenized = self.tokenizer(
                    text_inputs,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt"
                ).to(inputs.device)
                input_ids = tokenized["input_ids"]

                bos_token_id = self.tokenizer.bos_token_id #or self.tokenizer.eos_token_id
                bos_embedding = self.llm_backbone.get_input_embeddings()(
                    input_ids)#, device=features.device)
                  # shape: [1, 1, D]
                bos_embedding = bos_embedding.expand(features.size(0), -1, -1)  # [B, 1, D]

                # 2. Concatenate BOS token after vision features
                inputs_embeds = torch.cat([features, bos_embedding], dim=1)

                # 3. Build attention mask
                attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(inputs_embeds.device)



                #inputs_embeds = features
                #attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(inputs_embeds.device)

                generated_ids = self.llm_backbone.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_length=30,
                    pad_token_id=self.tokenizer.eos_token_id  # ensure proper stopping
                )
                #generated_ids = self.llm_backbone.generate(inputs_embeds=features, max_length=30)
                decoded_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                # Inference mode: generate captions
                return self.llm_backbone.generate(inputs_embeds=features, max_length=30),decoded_texts


        else:
            if self.use_text:
                
                tokenized = self.tokenizer(text_inputs,padding="longest",truncation=True,return_tensors="pt").to(inputs.device)
                text_embeds = self.llm_backbone.get_input_embeddings()(tokenized["input_ids"])
                
                features = torch.cat([text_embeds,features], dim=1)


            cls_tok = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
            features = torch.cat([cls_tok, features], dim=1)
            if features.size(1) > 512:
                features = features[:, :512, :]
            
            
            outputs = self.llm_backbone(inputs_embeds=features)
            
            outputs = outputs.last_hidden_state[:, 0]  # CLS token
            return self.classifier(outputs)





def apply_lora_to_backbone(backbone, target_modules,r=8, alpha=16, dropout=0.1):
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules
    )
    return get_peft_model(backbone, peft_config)

def freeze_all_except_lora(model):
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

def build_vision_model(model_name, target_modules,num_classes, use_lora=False, freeze_backbone=False):
    model = VideoClassificationModel(model_name=model_name, num_classes=num_classes, freeze_backbone=False)
    

    if use_lora:
        model.backbone = apply_lora_to_backbone(model.backbone,target_modules)
        freeze_all_except_lora(model.backbone)
    elif freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model,model.hidden_size

def build_vlm_model(vision_model,llm_model,num_classes,vision_target_modules,llm_target_modules,captioning=False,use_text=False,use_lora_vison=False,use_lora_llm=False,freeze_visionbackbone=False,freeze_llmbackbone=False):
    model = VisionLanguageModel(vision_model=vision_model,llm_model=llm_model,num_classes=num_classes,vision_target_modules=vision_target_modules,llm_target_modules=llm_target_modules,captioning=captioning,use_text=use_text,use_lora_vison=use_lora_vison,freeze_visionbackbone=freeze_visionbackbone,freeze_llmbackbone=freeze_llmbackbone)
    

    if use_lora_llm:
        model.llm_backbone = apply_lora_to_backbone(model.llm_backbone,llm_target_modules)
        freeze_all_except_lora(model.llm_backbone)
    elif freeze_llmbackbone:
        for param in model.llm_backbone.parameters():
            param.requires_grad = False

    return model
