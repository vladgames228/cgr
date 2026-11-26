import hashlib
import os
from pathlib import Path
from urllib.request import urlopen


MODEL_FILES = {
    'model.onnx': '98a01d88b7de996cdea58c32ca71208c09968d143798814b2ea09d3439dc334f',
    'tokenizer.json': '2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8',
}
MODEL_BASE = ('https://huggingface.co/sentence-transformers/'
              'paraphrase-multilingual-MiniLM-L12-v2/resolve/'
              'e8f8c211226b894fcb81acc59f3b34ba3efd5f42/')


def ensure_model(root):
    """Keep verified weights in the persistent data volume, including across rebuilds."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in MODEL_FILES.items():
        path = root / name
        if path.exists():
            existing = hashlib.sha256()
            with path.open('rb') as handle:
                while block := handle.read(1024 * 1024):
                    existing.update(block)
            if existing.hexdigest() == expected:
                continue
        temporary = root / (name + '.part')
        temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        try:
            remote = 'onnx/model_quint8_avx2.onnx' if name == 'model.onnx' else name
            with urlopen(MODEL_BASE + remote, timeout=60) as response, temporary.open('wb') as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f'Контрольная сумма {name} не совпала')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return root


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
        try:
            self.session = onnxruntime.InferenceSession(str(model), sess_options=options,
                                                        providers=['CPUExecutionProvider'])
        except Exception as exc:
            raise RuntimeError(f'ONNX Runtime не загрузил MiniLM: {exc}') from exc
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
