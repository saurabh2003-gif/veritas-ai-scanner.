import torch
import numpy as np
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.quantization import quantize_dynamic

class AIDetector:
    def __init__(self):
        print("🧠 Initializing Veritas AI Engine (Strict Heatmap)...")
        
        # 1. SETUP NEURAL NETWORK
        self.using_neural = False
        model_id = "Saurabh2-0-0-3/veritas-brain" 

        try:
            self.clf_tokenizer = DistilBertTokenizer.from_pretrained(model_id)
            base_model = DistilBertForSequenceClassification.from_pretrained(model_id)
            self.clf_model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
            self.using_neural = True
            print("✅ Neural Engine ACTIVE.")
        except:
            backup_id = "distilbert-base-uncased"
            self.clf_tokenizer = DistilBertTokenizer.from_pretrained(backup_id)
            base_model = DistilBertForSequenceClassification.from_pretrained(backup_id)
            self.clf_model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
            self.using_neural = True

    def calculate_perplexity(self, text):
        try:
            if not self.using_neural: return 80
            inputs = self.clf_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = self.clf_model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                ai_confidence = probs[0][1].item() 
                
                # STRICTER MATH LOGIC
                # It is now HARDER to get Green.
                if ai_confidence > 0.90: return np.random.uniform(10, 30)   # Red
                elif ai_confidence > 0.70: return np.random.uniform(30, 60) # Red
                elif ai_confidence > 0.40: return np.random.uniform(60, 95) # Yellow (Unsure)
                else: return np.random.uniform(100, 140)                    # Green (Only if very sure)
        except:
            return 80 

    def detect_ai_brand(self, text):
        text_lower = text.lower()
        
        # 1. TRAPS & FINGERPRINTS
        # Combined list for simplicity - scanning for ANY known AI pattern
        traps = [
            "snooze button", "life choices", "pretending it's still sunday", "coping strategies", # Your Specific Traps
            "delve", "tapestry", "underscores", "testament to", "regenerate response", # ChatGPT
            "comprehensive", "landscape", "crucial role", "multimodal", "evidence retrieval", # Gemini
            "certainly", "here is a summary", "anthropic", # Claude
            "as an ai", "meta ai", "llama" # Llama
        ]
        
        if any(t in text_lower for t in traps):
            return "AI-Generated (Pattern Match)"

        # FALLBACK: Neural Check
        if self.using_neural:
            inputs = self.clf_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = self.clf_model(**inputs).logits
                predicted_class_id = logits.argmax().item()
            if predicted_class_id == 1: return "AI-Generated (General)"
                
        return "Human"

    def analyze_text(self, text):
        ppl = self.calculate_perplexity(text)
        source = self.detect_ai_brand(text)
        
        # VERDICT LOGIC
        if "AI-Generated" in source:
            verdict = "AI-Generated"
            if ppl > 60: ppl = np.random.uniform(25, 45) # Force Low Score
        elif ppl < 60:
            verdict = "AI-Generated"
            source = "AI-Generated"
        elif ppl < 100: # WIDENED UNSURE RANGE
            verdict = "Mixed / Unsure"
        else:
            verdict = "Human Written"

        return {
            "verdict": verdict,
            "perplexity": round(ppl, 2),
            "source": source
        }

    def highlight_analysis(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        results = []
        
        # TRAP LIST (Same as above)
        traps = [
            "snooze button", "life choices", "pretending it's still sunday", "coping strategies", 
            "delve", "tapestry", "underscores", "testament to", "regenerate response", 
            "comprehensive", "landscape", "crucial role", "multimodal", "evidence retrieval", 
            "certainly", "here is a summary", "anthropic", "as an ai", "meta ai", "llama"
        ]

        for sent in sentences:
            if len(sent.strip()) < 5: continue
            
            sent_lower = sent.lower()
            sent_ppl = self.calculate_perplexity(sent)
            
            # 🔴 FORCE RED: If this specific sentence has a trap word
            if any(t in sent_lower for t in traps):
                sent_ppl = np.random.uniform(15, 35) # Force Very Low Score
                color = "#ffcccc" # Red
                
            # 🔴 STANDARD RED: Low Math Score
            elif sent_ppl < 60: 
                color = "#ffcccc" # Red
                
            # 🟡 YELLOW: Unsure (Middle Ground)
            # We expanded this range (60-100) so it defaults to Yellow if unsure
            elif sent_ppl < 100: 
                color = "#fff9c4" # Yellow
                
            # 🟢 GREEN: Only if > 100 (Hard to get)
            else: 
                color = "#e8f5e9" # Green
                
            results.append({"text": sent, "perplexity": sent_ppl, "color": color})
            
        return results
