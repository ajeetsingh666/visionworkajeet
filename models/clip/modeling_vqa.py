import torch
from transformers import AutoTokenizer, AutoModelForVisualQuestionAnswering
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class VQAViolationDetection:
    def __init__(self, model_name: str = 'Qwen/Qwen2-VL-2B-Instruct', device="cpu"):
        """
        Initializes the ViolationDetection object.
        
        :param frame_extractor: An instance of the frame extraction class from another module.
        :param model_name: Pretrained model name for VQA.
        """
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto")
        

    def predict(self, image, query):
        """
        Detect violation in a given frame based on the query using VQA model.
        
        :param frame: Image (frame) from the video.
        :param query: Query string to ask the VQA model.
        :return: Response from the VQA model.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

        # Convert frame to RGB and preprocess for the model
        text = self.processor.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

