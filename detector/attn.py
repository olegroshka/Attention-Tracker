import numpy as np
from .utils import process_attn, calc_attn_score


class AttentionDetector():
    def __init__(self, model, pos_examples=None, neg_examples=None, use_token="first", instruction="Say xxxxxx",
                 threshold=0.5, max_generated_tokens=50):  # Added max_generated_tokens
        self.name = "attention"
        self.attn_func = "normalize_sum"
        self.model = model
        self.important_heads = model.important_heads
        self.instruction = instruction
        self.use_token = use_token
        self.threshold = threshold
        self.max_generated_tokens = max_generated_tokens  # Store it
        # The __init__ method uses self.model.query, which returns 6 values.
        # We'll assume self.model.inference in detect also has a similar signature
        # where the first value could be the generated text.
        if pos_examples and neg_examples:
            pos_scores, neg_scores = [], []
            for prompt in pos_examples:
                # Assuming the first return value of query might be generated text, though it's ignored here.
                _, _, attention_maps, _, input_range, _ = self.model.query(
                    prompt, return_type="attention")
                pos_scores.append(self.attn2score(attention_maps, input_range))

            for prompt in neg_examples:
                _, _, attention_maps, _, input_range, _ = self.model.query(
                    prompt, return_type="attention")
                neg_scores.append(self.attn2score(attention_maps, input_range))

            if neg_scores:  # Ensure neg_scores is not empty
                self.threshold = np.mean(neg_scores)
            elif pos_scores:  # Fallback if only pos_scores are available
                self.threshold = np.mean(pos_scores) - 4 * np.std(pos_scores)


        elif pos_examples and not neg_examples:  # Corrected logic for this block
            pos_scores = []
            for prompt in pos_examples:
                _, _, attention_maps, _, input_range, _ = self.model.query(
                    prompt, return_type="attention")
                pos_scores.append(self.attn2score(attention_maps, input_range))

            if pos_scores:  # Ensure pos_scores is not empty
                self.threshold = np.mean(pos_scores) - 4 * np.std(pos_scores)

    def attn2score(self, attention_maps, input_range):
        if self.use_token == "first":
            # Ensure attention_maps is not empty and is a list/tuple before slicing
            if attention_maps and isinstance(attention_maps, (list, tuple)):
                attention_maps = [attention_maps[0]]
            # If it's a single map not in a list, wrap it
            elif attention_maps and not isinstance(attention_maps, (list, tuple)):
                attention_maps = [attention_maps]

        scores = []
        if attention_maps:  # Check if attention_maps is not None or empty
            for attention_map in attention_maps:
                if attention_map is None:  # Skip if an individual map is None
                    continue
                # Ensure input_range is valid before processing
                if input_range is None:
                    print("Warning: input_range is None in attn2score. Skipping this attention map.")
                    continue
                heatmap = process_attn(
                    attention_map, input_range, self.attn_func)
                score = calc_attn_score(heatmap, self.important_heads)
                scores.append(score)

        return sum(scores) if len(scores) > 0 else 0

    def detect(self, data_prompt):
        # Assuming the first return value of self.model.inference is the generated text.
        # The original call was: _, _, attention_maps, _, input_range, _
        # We change it to capture the generated text.
        # The number of return values from self.model.inference needs to be known.

        generated_text = "N/A (inference not attempted or failed early)"
        attention_maps = None
        input_range = None

        try:
            # Attempt to unpack assuming generated_text is the first element.
            # The exact number of return values from self.model.inference is crucial.
            inference_outputs = self.model.inference(
                self.instruction,
                data_prompt,
                max_output_tokens=self.max_generated_tokens  # Use the configurable max_generated_tokens
            )

            # It's safer to check the type of inference_outputs and access elements accordingly.
            # Assuming self.model.inference returns a tuple where:
            # element 0: generated_text
            # element 2: attention_maps (if 6 elements total)
            # element 4: input_range (if 6 elements total)
            # This structure is based on the self.model.query in __init__
            # You MUST verify the actual return signature of your self.model.inference method.

            if isinstance(inference_outputs, tuple):
                if len(inference_outputs) >= 1:  # At least generated_text should be there
                    generated_text = inference_outputs[0]

                # Attempt to get attention_maps and input_range based on expected length
                # This part is highly dependent on your model.inference() structure
                if len(inference_outputs) >= 6:  # Assuming 6-element tuple like model.query
                    attention_maps = inference_outputs[2]
                    input_range = inference_outputs[4]
                elif len(inference_outputs) >= 3:  # A more minimal assumption: text, attentions, input_range
                    # This is a guess, adjust if your model.inference returns differently
                    # e.g., if it's (text, attentions, input_range, ...)
                    if len(inference_outputs) > 1: attention_maps = inference_outputs[1]
                    if len(inference_outputs) > 2: input_range = inference_outputs[2]
                    if len(inference_outputs) < 6:  # Print warning only if it's not the 6-element case
                        print(
                            f"Warning: self.model.inference returned {len(inference_outputs)} values. Mapping for attention_maps/input_range might be a guess.")
                else:
                    print(
                        f"Warning: self.model.inference returned an unexpected number of values ({len(inference_outputs)}). Cannot reliably extract attention_maps/input_range.")
            else:
                print(
                    f"Warning: self.model.inference did not return a tuple as expected. Got {type(inference_outputs)}. Cannot extract details.")
                generated_text = "N/A (unexpected inference output type)"


        except Exception as e:
            print(f"Error during model inference or unpacking results: {e}")
            # generated_text might have been set before the error, or keep it as N/A
            if generated_text.startswith("N/A"):  # If not already set from a partial success
                generated_text = f"N/A (error in inference: {e})"
            # attention_maps and input_range remain None or their last state

        focus_score = self.attn2score(attention_maps, input_range)

        # Include the captured generated_text in the returned dictionary
        return bool(focus_score <= self.threshold), {
            "focus_score": focus_score,
            "model_generated_text": generated_text
        }

