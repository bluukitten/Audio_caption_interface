from __future__ import annotations
import argparse

import pandas as pd
from pathlib import Path
import soundfile
import torch
import librosa
from audidata.datasets import Clotho
from audidata.io.crops import RandomCrop
from audidata.transforms import Mono

from data.text_normalization import TextNormalization
from data.text_tokenization import BertTokenizer
from train import get_audio_encoder, get_llm_decoder, get_audio_latent


'''
def inference(args):

    # Arguments
    ckpt_path = args.ckpt_path

    # Default parameters
    sr = 32000  # To be consistent with the encoder
    device = "cuda"
    split = "test"
    max_length = 30  # Max caption length
    clip_duration = 10.  # Audio clip duration
    audio_encoder_name = "Cnn14"
    llm_decoder_name = "Llama"
    num_samples = 3
    temperature = 1.0
    top_k = 200

    # Dataset
    root = "/datasets/clotho"

    # Audio Cropper
    crop = RandomCrop(clip_duration=clip_duration, end_pad=0.)

    # Caption transforms
    target_transform = [
        TextNormalization(),  # Remove punctuations
        BertTokenizer(max_length=max_length)  # Convert captions to token IDs
    ]
    tokenizer = target_transform[1].tokenizer
    start_token_id = tokenizer.cls_token_id  # 101
    text_vocab_size = tokenizer.vocab_size  # 30,522

    # Dataset
    test_dataset = Clotho(
        root=root,
        split=split,
        sr=sr,
        crop=crop,
        transform=Mono(),
        target_transform=target_transform
    )

    # Load audio encoder
    audio_encoder, audio_latent_dim = get_audio_encoder(model_name=audio_encoder_name)
    audio_encoder.to(device)

    # Load LLM decoder
    llm_decoder = get_llm_decoder(
        model_name=llm_decoder_name, 
        audio_latent_dim=audio_latent_dim, 
        text_vocab_size=text_vocab_size
    )
    llm_decoder.load_state_dict(torch.load(ckpt_path))
    llm_decoder.to(device)

    text_ids = torch.LongTensor([[start_token_id]]).to(device)  # (b, 1)

    for audio_idx, data in enumerate(test_dataset):

        # Move data to device
        audio = torch.Tensor(data["audio"][None, :, :]).to(device)  # shape: (b, c, t_audio)
        path = data["audio_path"]
        caption = data["caption"]

        print("------------")
        print("Audio path: {}".format(path))
        print("Ground truth: {}".format(caption))
        soundfile.write(file="_zz.wav", data=audio[0][0].cpu().numpy(), samplerate=sr)
        
        # Extract audio embeddings
        audio_latent = get_audio_latent(
            model_name=audio_encoder_name, 
            model=audio_encoder, audio=audio
        )

        # Sample    
        for n in range(num_samples):

            # Combine audio embeddings and text ids
            input_seqs = [audio_latent, text_ids]
            seq_types = ["audio", "text"]

            with torch.no_grad():
                llm_decoder.eval()
            
                outputs = llm_decoder.generate(
                    seqs=input_seqs,
                    seq_types=seq_types,
                    max_new_tokens=max_length, 
                    temperature=temperature, 
                    top_k=top_k
                )
                # list of Tensor

            sampled_text_ids = outputs[-1][0].cpu().numpy()
            strings = tokenizer.decode(token_ids=sampled_text_ids, skip_special_tokens=True)
            print(strings)
            
        if audio_idx == 10:
            break
'''

def inference(audio_path: str):

    # Arguments
    ckpt_path = '/Users/huiyufei/Desktop/mini_audio_caption/checkpoints/train/Cnn14_Llama/step=20000.pth'

    # Default parameters
    sr = 32000  # To be consistent with the encoder
    device = "cpu"
    #split = "test"
    max_length = 30  # Max caption length
    clip_duration = 10.  # Audio clip duration
    audio_encoder_name = "Cnn14"
    llm_decoder_name = "Llama"
    num_samples = 1
    temperature = 1.0
    top_k = 200

    # Dataset
    #root = args.data_dir if hasattr(args, 'data_dir') else "/Users/huiyufei/datasets/clotho"


    # Audio Cropper
    crop = RandomCrop(clip_duration=clip_duration, end_pad=0.)

    # Caption transforms
    target_transform = [
        TextNormalization(),  # Remove punctuations
        BertTokenizer(max_length=max_length)  # Convert captions to token IDs
    ]
    tokenizer = target_transform[1].tokenizer
    start_token_id = tokenizer.cls_token_id  # 101
    text_vocab_size = tokenizer.vocab_size  # 30,522

    # # Dataset
    #meta_dict = get_clotho_meta(root, split)
    #audios_num = len(meta_dict["audio_name"])

    # Load audio encoder
    audio_encoder, audio_latent_dim = get_audio_encoder(model_name=audio_encoder_name)
    audio_encoder.to(device)

    # Load LLM decoder
    llm_decoder = get_llm_decoder(
        model_name=llm_decoder_name, 
        audio_latent_dim=audio_latent_dim, 
        text_vocab_size=text_vocab_size
    )
    llm_decoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    llm_decoder.to(device)

    # Text start token
    text_ids = torch.LongTensor([[start_token_id]]).to(device)  # (b, 1)

    #for audio_idx in range(audios_num):

    #audio_path = meta_dict["audio_path"][audio_idx]
    #captions = meta_dict["captions"][audio_idx]

    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)

        # Move data to device
    audio = torch.Tensor(audio[None, None, :]).to(device)  # shape: (b, c, t_audio)
        
       # print("------------")
        #print("Audio path: {}".format(audio_path))
       # for caption in captions:
        #    print("Ground truth: {}".format(caption))
        # soundfile.write(file="_zz.wav", data=audio[0][0].cpu().numpy(), samplerate=sr)
        
        # Extract audio embeddings
    audio_latent = get_audio_latent(
            model_name=audio_encoder_name, 
            model=audio_encoder, audio=audio
        )

        # Sample    
    result = None
    for n in range(num_samples):
        # Combine audio embeddings and text ids
        input_seqs = [audio_latent, text_ids]
        seq_types = ["audio", "text"]

        with torch.no_grad():
            llm_decoder.eval()
            outputs = llm_decoder.generate(
                seqs=input_seqs,
                seq_types=seq_types,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k
            )
        sampled_text_ids = outputs[-1][0].cpu().numpy()
        strings = tokenizer.decode(token_ids=sampled_text_ids, skip_special_tokens=True)
        print("Prediction: {}".format(strings))
        result = strings  # 保存结果
        break  # 只取第一个
    return result


def get_clotho_meta(root: str, split: str) -> dict:
    r"""Load Clotho audio paths and captions."""
    if split == "train":
        meta_csv = Path(root, "clotho_captions_development.csv")
        audios_dir = Path(root, "clotho_audio_development")

    elif split == "test":
        meta_csv = Path(root, "clotho_captions_evaluation.csv")
        audios_dir = Path(root, "clotho_audio_evaluation")

    else:
        raise ValueError(split)

    meta_dict = {
        "audio_name": [],
        "audio_path": [],
        "captions": []
    }

    df = pd.read_csv(meta_csv, sep=',')

    for n in range(len(df)):
        meta_dict["audio_name"].append(df["file_name"][n])
        meta_dict["audio_path"].append(Path(audios_dir, df["file_name"][n]))
        meta_dict["captions"].append([df["caption_{}".format(i)][n] for i in range(1, 6)])

    return meta_dict


def tokens_to_string(tokens, tokenizer):
    return "".join([tokenizer.itos(token) for token in tokens])


if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--ckpt_path', type=str, required=True)
    #parser.add_argument('--data_dir', type=str, default="/Users/huiyufei/datasets/clotho")
    audio_path = '/Users/huiyufei/datasets/clotho/development/_01storm - orage.wav'  # Example audio path, not used in this script

    inference(audio_path)