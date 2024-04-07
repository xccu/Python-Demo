#https://paddlehub.readthedocs.io/zh_CN/release-v2.1/finetune/sequence_labeling.html

import os

import paddle

os.environ['HUB_HOME'] = "./modules"
import paddlehub as hub

model = hub.Module(name="chinese_ocr_db_crnn_server")
train_dataset = hub.datasets.LCQMC(tokrnizer=model.get_tokenizer(), mode='train')
dev_dataset = hub.datasets.LCQMC(mode='dev')
#train_dataset = hub.datasets.LCQMC(tokenizer=model.get_name_prefix(), max_seq_len=128, mode='train')
#dev_dataset = hub.datasets.LCQMC(tokenizer=model.get_name_prefix(), max_seq_len=128, mode='dev')
#test_dataset = hub.datasets.LCQMC(tokenizer=model.get_name_prefix(), max_seq_len=128, mode='test')

optimizer = paddle.optimizer.AdamW(learning_rate=5e-5, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='./', use_gpu=True)


trainer.train(
    train_dataset,
    epochs=10,
    batch_size=32,
    eval_dataset=dev_dataset,
    save_interval=2,
)
#trainer.evaluate(test_dataset, batch_size=32)


