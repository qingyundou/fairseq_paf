# import torch
# import fairseq

# # List available models
# torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]

# # Load a transformer trained on WMT'16 En-De
# # Note: WMT'19 models use fastBPE instead of subword_nmt, see instructions below
# en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt')
# en2de.eval()  # disable dropout

# # The underlying model is available under the *models* attribute
# assert isinstance(en2de.models[0], fairseq.models.transformer.TransformerModel)

# # Move model to GPU for faster translation
# en2de.cuda()

# # Translate a sentence
# print(en2de.translate('Hello world!'))
# # 'Hallo Welt!'

# # Batched translation
# en2de.translate(['Hello world!', 'The cat sat on the mat.'])
# # ['Hallo Welt!', 'Die Katze saß auf der Matte.']


# ------------------------------------


from fairseq.models.transformer import TransformerModel
zh2en = TransformerModel.from_pretrained(
  '/home/dawna/tts/qd212/models/fairseq/checkpoints/fconv/',
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path='data-bin/iwslt14.tokenized.de-en'
)
zh2en.translate('Hallo Welt!')
# 'Hello World'

