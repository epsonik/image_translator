from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import sys
import time

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from config_translator import general
import csv
from tensorflow.keras.utils import to_categorical


def calculate_results(expected, results, config):
    """
    Method to evaluate image captioning model by calculating scores:
     "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"m "METEOR", "ROUGE_L", "CIDEr", "WMD", "SPICE"

    Parameters
    ----------
    expected: dict
        Dictionary with ground truth captions, identidied by image_id
    results: dict
        Dictionary with predicted captions, identified by image_id
    config
        Configuration of training and testing
    Returns
    -------
    calculated_metrics
        Result of the metrics: "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"m "METEOR", "ROUGE_L", "CIDEr", "WMD", "SPICE"

    """
    sys.path.append(general["coco-caption_path"])
    from pycocoevalcap.eval_any import COCOEvalCap
    # Load expected captions(ground truth from dataset) and results(predicted captions for specific image)
    # to the evaluation framework
    cocoEvalObj = COCOEvalCap(expected, results)
    # Evaluate
    cocoEvalObj.evaluate()
    calculated_metrics = {}
    # Store metrics  values in dictionary by metrics names
    for metric, score in cocoEvalObj.eval.items():
        calculated_metrics[metric] = score
    print(calculated_metrics)
    print("Calculating final results")
    imgToEval = cocoEvalObj.imgToEval
    for p in results:
        print(imgToEval)
        image_id, caption = p, results[p][0]['caption']
        imgToEval[image_id]['caption'] = caption
        imgToEval[image_id]['ground_truth_captions'] = [x['caption'] for x in expected[p]]

    evaluation_results_save_path = os.path.join(general["results_directory"], config["data_name"] + '.json')
    print("Results saved to ")
    print(evaluation_results_save_path)
    # Path to save evaluation results
    with open(evaluation_results_save_path, 'w') as outfile:
        json.dump(
            {'overall': calculated_metrics, 'dataset_name': config["test"]["dataset_name"], 'imgToEval': imgToEval},
            outfile)
    return calculated_metrics


def greedySearch(photo, model, wordtoix, ixtoword, max_length):
    """
    Method to put ground truth captions and results to the structure accepted by evaluation framework

    Parameters
    ----------
    encoding_test: str
        Path to the results directory
    test_captions_mapping
        Dictionary with keys-image_id's , values -list of ground truth captions for specific image.
    wordtoix
        Dictionary with keys-words , values -id of word
    ixtoword
        Dictionary with keys-id of words , values -words
    max_length
        Max number of words in caption on dataset
    model
        Image captioning model to predict captions
    Returns
    -------
    expected: dict
        Dictionary with ground truth captions, identidied by image_id
    results: dict
        Dictionary with predicted captions, identified by image_id

    """
    in_text = general["START"]
    for i in range(max_length):
        # Get previously generated sequence
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        # Pad sewuences to the maximum length
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Predict sequence with the learned model
        yhat = model.predict([photo, sequence], verbose=0)
        # Get word with the highest propability
        yhat = np.argmax(yhat)
        # Transform index of word to the word by previously created dictionary
        word = ixtoword[yhat]
        in_text += ' ' + word
        # When we achieve STOP word, sequence is generated
        if word == general["STOP"]:
            break
    final = in_text.split()
    # remove start and stop words
    final = final[1:-1]
    # Create sencece by joining tokens
    final = ' '.join(final)
    return final


def prepare_for_evaluation(data, model, decoder_model):
    """
    Method to put ground truth captions and results to the structure accepted by evaluation framework

    Parameters
    ----------
    encoding_test: str
        Path to the results directory
    test_captions_mapping
        Dictionary with keys-image_id's , values -list of ground truth captions for specific image.
    wordtoix
        Dictionary with keys-words , values -id of word
    ixtoword
        Dictionary with keys-id of words , values -words
    max_length
        Max number of words in caption on dataset
    model
        Image captioning model to predict captions
    images_processor

    Returns
    -------
    expected: dict
        Dictionary with ground truth captions, identidied by image_id
    results: dict
        Dictionary with predicted captions, identified by image_id

    """
    # Get all image-ids from test dataset
    expected = dict()
    results = dict()
    print("Preparing for evaluation")
    # calculation of metrics for test images dataset
    index = 0
    for pair in data.test_dataset:
        image_id = pair["image_id"]
        expected[image_id] = []

        # Put ground truth captions to the structure accepted by evaluation framework.
        for desc in pair["captions"]:
            expected[image_id].append({"image_id": image_id, "caption": desc})
        # Predict captions

        st = time.time()
        generated = translate_sentence(pair["bbox_categories"], data.word2idx_outputs, data.max_output_length, model,
                                       decoder_model)
        print(generated)
        et = time.time()
        # get the execution time
        elapsed_time = et - st
        print('Execution time:', elapsed_time * 1000, 'miliseconds')

        # get the execution time
        # Put predicted captions to the structure accepted by evaluation framework.
        results[image_id] = [{"image_id": image_id, "caption": generated, "time": elapsed_time}]
        if index % 100 == 0:
            print("Processed:")
            print(index)
            print("saved")
        index += 1
    return expected, results


def encode_sequences(tokenizer, length, lines, padding_type='post'):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen=length, padding=padding_type)
    return X


def encode_output(sequences, vocab_size):
    ylist = list()
    for seq in sequences:
        encoded = to_categorical(seq, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape((sequences.shape[0], sequences.shape[1], vocab_size))
    return y


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def translate_sentence(input_seq, word2idx_outputs, max_out_len, encoder_model, decoder_model):
    idx2word_target = {v: k for k, v in word2idx_outputs.items()}
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs[general['START']]
    eos = word2idx_outputs[general['STOP']]
    output_sentence = []

    for _ in range(max_out_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)


def generate_report(results_path):
    """
    Method to generate summary of the test results. Made from files in the results directory.

    Parameters
    ----------
    results_path: str
        Path to the results directory
    Returns
    -------
        CSV file with summary of the results.

    """
    # Names of the evaluation metrics
    header = ["config_name", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE", "WMD"]
    print(f'\n Final results saved to final_results.csv')
    all_results = []
    # iterate over all files in results directory
    for x in os.listdir(results_path):
        # use just .json files
        if x.endswith(".json"):
            # Load data from file with results particular for configuaration
            results_for_report = json.load(open("./" + results_path + "/" + x, 'r'))
            # Add column with the configuration name to name the specific results.
            results_for_report["overall"]["config_name"] = x.split(".")[0]
            # Save the results to the table to save it in the next step
            all_results.append(results_for_report["overall"])
    # Save final csv file
    with open("./" + results_path + "/final_results.csv", 'w') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_results)
