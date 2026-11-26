import hashlib
from pathlib import Path


class Encoder:
    """Optional local ONNX sentence encoder with masked mean pooling."""

    def __init__(self, model_dir):
        import numpy
        import onnxruntime
        from tokenizers import Tokenizer

        root = Path(model_dir)
        model = root / 'model.onnx'
        tokenizer = root / 'tokenizer.json'
        digest = hashlib.sha256()
        for path in (model, tokenizer):
            with path.open('rb') as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
        self.identity = 'minilm-v1-' + digest.hexdigest()
        self.np = numpy
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        pad = '<pad>'
        pad_id = self.tokenizer.token_to_id(pad)
        if pad_id is None:
            raise ValueError('В словаре ONNX-модели отсутствует <pad>')
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(str(model), sess_options=options,
                                                    providers=['CPUExecutionProvider'])
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.output = 'last_hidden_state'
        if self.output not in {item.name for item in self.session.get_outputs()}:
            raise ValueError('ONNX-модель должна возвращать last_hidden_state')

    def encode(self, texts):
        np = self.np
        for start in range(0, len(texts), 8):
            batch = self.tokenizer.encode_batch(texts[start:start + 8])
            mask = np.asarray([item.attention_mask for item in batch], dtype=np.int64)
            inputs = {
                'input_ids': np.asarray([item.ids for item in batch], dtype=np.int64),
                'attention_mask': mask,
                'token_type_ids': np.asarray([item.type_ids for item in batch], dtype=np.int64),
            }
            vectors, = self.session.run([self.output], {key: inputs[key] for key in self.inputs})
            summed = (vectors * mask[:, :, None]).sum(axis=1)
            for vector in summed:
                scale = np.max(np.abs(vector))
                yield np.rint(vector * (127 / scale)).astype(np.int8).tolist() if scale else [0] * len(vector)
