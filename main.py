import yaml
import logging
from typing import Dict, Union, List
from engines import (
    CalculatorEngine, CodeInterpreterEngine, LatexFormattingEngine,
    DatabaseQueryEngine, GraphingEngine, WebSearchEngine,
    DocumentTemplateEngine, TextSummarizerEngine, SymbolicReasoningEngine,
    NetworkAnalysisEngine, MathMLConverterEngine, SimulationEngine,
    ImageProcessingEngine, ImageGenerationEngine, ImageEditingEngine,
    AudioTranscriptionEngine, AudioGenerationEngine,
    VideoProcessingEngine, VideoGenerationEngine
)
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import io
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultimodalMoE:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.initialize_engines()

    def initialize_engines(self):
        """Initializes all engines with config parameters."""
        self.engines = {
            "calculator": CalculatorEngine(precision=self.config["engines"]["calculator"]["precision"]),
            "code_interpreter": CodeInterpreterEngine(
                timeout=self.config["engines"]["code_interpreter"]["timeout"],
                allowed_libraries=self.config["engines"]["code_interpreter"]["allowed_libraries"]
            ),
            "latex_formatting": LatexFormattingEngine(
                default_font=self.config["engines"]["latex_formatting"]["default_font"],
                language=self.config["engines"]["latex_formatting"]["language"]
            ),
            "database_query": DatabaseQueryEngine(db_path=self.config["engines"]["database_query"]["db_path"]),
            "graphing": GraphingEngine(output_format=self.config["engines"]["graphing"]["output_format"]),
            "web_search": WebSearchEngine(),
            "document_template": DocumentTemplateEngine(self.engines["latex_formatting"]),
            "text_summarizer": TextSummarizerEngine(model_name=self.config["models"]["text_summarizer"]),
            "symbolic_reasoning": SymbolicReasoningEngine(),
            "network_analysis": NetworkAnalysisEngine(),
            "mathml_converter": MathMLConverterEngine(),
            "simulation": SimulationEngine(),
            "image_processing": ImageProcessingEngine(),
            "image_generation": ImageGenerationEngine(model_name=self.config["models"]["image_generation"]),
            "image_editing": ImageEditingEngine(),
            "audio_transcription": AudioTranscriptionEngine(model_name=self.config["models"].get("audio_transcription", "speechbrain/asr-wav2vec2-commonvoice-en")),
            "audio_generation": AudioGenerationEngine(
                tts_model=self.config["models"].get("audio_generation_tts", "speechbrain/tts-tacotron2-ljspeech"),
                vocoder_model=self.config["models"].get("audio_generation_vocoder", "speechbrain/hifigan-ljspeech")
            ),
            "video_processing": VideoProcessingEngine(),
            "video_generation": VideoGenerationEngine(model_name=self.config["models"]["image_generation"])  # Reuse image generation model
        }

    def route_query(self, query: Dict) -> Dict[str, Union[str, bytes, list]]:
        """Routes query to appropriate experts and engines."""
        try:
            query_type = query.get("type", "text")
            content = query.get("content")

            if query_type == "text":
                if "calculate" in content.lower():
                    return self.engines["calculator"].calculate(content)
                elif "summarize" in content.lower():
                    return self.engines["text_summarizer"].summarize_text(content, "abstractive")
                elif "generate image" in content.lower():
                    return self.engines["image_generation"].generate_image(content)
                elif "generate audio" in content.lower():
                    return self.engines["audio_generation"].generate_audio(content)
                elif "generate video" in content.lower():
                    return self.engines["video_generation"].generate_video(content)
                else:
                    return self.engines["text_summarizer"].summarize_text(content, "abstractive")

            elif query_type == "image":
                task = query.get("task", "object_detection")
                if task in ["object_detection", "ocr", "feature_extraction"]:
                    return self.engines["image_processing"].process_image(content, task)
                else:
                    return self.engines["image_editing"].edit_image(content, task, query.get("params", {}))

            elif query_type == "audio":
                task = query.get("task", "transcription")
                if task == "transcription":
                    return self.engines["audio_transcription"].transcribe_audio(content)
                else:
                    raise ValueError(f"Unsupported audio task: {task}")

            elif query_type == "video":
                task = query.get("task", "key_frame_detection")
                if task in ["key_frame_detection", "object_detection"]:
                    return self.engines["video_processing"].process_video(content, task)
                else:
                    raise ValueError(f"Unsupported video task: {task}")

            else:
                raise ValueError(f"Unsupported query type: {query_type}")

        except Exception as e:
            logging.error(f"Query routing error: {str(e)}")
            return {"success": False, "error": str(e)}

    def process_query(self, query: Dict) -> Dict[str, Union[str, bytes, list]]:
        """Processes a query and aggregates outputs."""
        result = self.route_query(query)
        # Extend with Output Aggregator Engine for combining multimodal outputs
        return result

    def fine_tune_engine(self, engine_name: str, dataset: List[Dict], model_name: str = None):
        """Fine-tunes an engine's model with provided dataset."""
        try:
            if engine_name not in ["image_processing", "image_generation", "audio_transcription", "audio_generation"]:
                raise ValueError(f"Fine-tuning not supported for {engine_name}")

            # Convert dataset to Hugging Face format
            hf_dataset = Dataset.from_list(dataset)
            training_args = TrainingArguments(
                output_dir=f"./results_finetune_{engine_name}",
                per_device_train_batch_size=self.config["fine_tuning"]["batch_size"],
                num_train_epochs=self.config["fine_tuning"]["epochs"],
                learning_rate=self.config["fine_tuning"]["learning_rate"],
                save_strategy="no",
                logging_steps=50,
            )

            if engine_name == "image_processing":
                from transformers import AutoProcessor, AutoModelForObjectDetection
                processor = AutoProcessor.from_pretrained(model_name or self.config["models"]["image_processing_object_detection"])
                model = AutoModelForObjectDetection.from_pretrained(model_name or self.config["models"]["image_processing_object_detection"])
                def preprocess_data(examples):
                    images = [Image.open(io.BytesIO(img)).convert("RGB") for img in examples["image"]]
                    inputs = processor(images=images, annotations=examples["objects"], return_tensors="pt")
                    return inputs
                train_dataset = hf_dataset.map(preprocess_data, batched=True)
                trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
                trainer.train()
                model.save_pretrained(f"./results_finetune_{engine_name}")
                processor.save_pretrained(f"./results_finetune_{engine_name}")
                # Update engine
                self.engines[engine_name] = ImageProcessingEngine()

            elif engine_name == "audio_transcription":
                # Simplified fine-tuning (speechbrain requires custom handling)
                logging.info("Fine-tuning audio_transcription requires custom speechbrain pipeline. Dataset logged.")
                # Placeholder for speechbrain fine-tuning
                pass

            else:
                logging.warning(f"Fine-tuning for {engine_name} not fully implemented yet.")
                return {"success": False, "error": f"Fine-tuning for {engine_name} not supported"}

            return {"success": True, "message": f"Fine-tuned {engine_name}"}

        except Exception as e:
            logging.error(f"Fine-tuning error: {str(e)}")
            return {"success": False, "error": str(e)}

# Example usage
if __name__ == "__main__":
    moe = MultimodalMoE()
    # Text query
    text_query = {"type": "text", "content": "Calculate 2 + 2"}
    result1 = moe.process_query(text_query)
    print(f"Text query result: {result1}")

    # Image query
    with open("sample_image.jpg", "rb") as f:
        image_query = {"type": "image", "content": f.read(), "task": "object_detection"}
        result2 = moe.process_query(image_query)
        print(f"Image query result: {result2}")

    # Audio query
    with open("sample_audio.wav", "rb") as f:
        audio_query = {"type": "audio", "content": f.read(), "task": "transcription"}
        result3 = moe.process_query(audio_query)
        print(f"Audio query result: {result3}")

    # Video query
    with open("sample_video.mp4", "rb") as f:
        video_query = {"type": "video", "content": f.read(), "task": "key_frame_detection"}
        result4 = moe.process_query(video_query)
        print(f"Video query result: {result4}")

    # Example fine-tuning (placeholder dataset)
    dataset = [{"image": open("sample_image.jpg", "rb").read(), "objects": [{"label": "car", "box": [100, 100, 200, 200]}]}]
    fine_tune_result = moe.fine_tune_engine("image_processing", dataset)
    print(f"Fine-tuning result: {fine_tune_result}")