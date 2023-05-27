import itertools
from math import sqrt

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from collections import defaultdict
from itertools import combinations
import sys

num_class = 10


def generate_columns(inpt, block_size):
    final_output = []
    batch = inpt.permute(0, 2, 3, 1)  # color channel last
    for pos in range(batch.shape[2]):

        out_c1 = torch.zeros(batch.shape).cuda()
        out_c2 = torch.zeros(batch.shape).cuda()
        if (pos + block_size > batch.shape[2]):
            out_c1[:, :, pos:] = batch[:, :, pos:]
            out_c2[:, :, pos:] = 1. - batch[:, :, pos:]

            out_c1[:, :, :pos + block_size - batch.shape[2]] = batch[:, :, :pos + block_size - batch.shape[2]]
            out_c2[:, :, :pos + block_size - batch.shape[2]] = 1. - batch[:, :, :pos + block_size - batch.shape[2]]
        else:
            out_c1[:, :, pos:pos + block_size] = batch[:, :, pos:pos + block_size]
            out_c2[:, :, pos:pos + block_size] = 1. - batch[:, :, pos:pos + block_size]

        out_c1 = out_c1.permute(0, 3, 1, 2)
        out_c2 = out_c2.permute(0, 3, 1, 2)
        out = torch.cat((out_c1, out_c2), 1)
        final_output.append(out)

    return final_output


def generate_rows(inpt, block_size):
    batch = inpt.permute(0, 2, 3, 1)  # color channel last
    final_output = []
    for pos in range(batch.shape[2]):
        out_c1 = torch.zeros(batch.shape).cuda()
        out_c2 = torch.zeros(batch.shape).cuda()
        if (pos + block_size > batch.shape[2]):
            out_c1[:, pos:, :] = batch[:, pos:, :]
            out_c2[:, pos:, :] = 1. - batch[:, pos:, :]

            out_c1[:, :pos + block_size - batch.shape[2], :] = batch[:, :pos + block_size - batch.shape[2], :]
            out_c2[:, :pos + block_size - batch.shape[2], :] = 1. - batch[:, :pos + block_size - batch.shape[2], :]
        else:
            out_c1[:, pos:pos + block_size, :] = batch[:, pos:pos + block_size, :]
            out_c2[:, pos:pos + block_size, :] = 1. - batch[:, pos:pos + block_size, :]

        out_c1 = out_c1.permute(0, 3, 1, 2)
        out_c2 = out_c2.permute(0, 3, 1, 2)
        out = torch.cat((out_c1, out_c2), 1)
        final_output.append(out)
        # plotimage = out_c1[0, :, :, :].data.cpu().numpy()
        # plt.imshow(np.transpose(plotimage, (1, 2, 0)), interpolation='nearest')
        #
        # # plotimage = np.transpose(plotimage, (1, 2, 0))
        # # plt.imshow(plotimage)
        # plt.show()
    return final_output


def generate_blocks(inpt, block_size):
    batch = inpt.permute(0, 2, 3, 1)  # color channel last
    final_output = []
    for xcorner in range(batch.shape[1]):
        for ycorner in range(batch.shape[2]):

            out_c1 = torch.zeros(batch.shape).cuda()
            out_c2 = torch.zeros(batch.shape).cuda()

            if (xcorner + block_size > batch.shape[1]):
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[:, xcorner:, ycorner:] = batch[:, xcorner:, ycorner:]
                    out_c2[:, xcorner:, ycorner:] = 1. - batch[:, xcorner:, ycorner:]

                    out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:] = batch[:,
                                                                                  :xcorner + block_size - batch.shape[
                                                                                      1], ycorner:]
                    out_c2[:, :xcorner + block_size - batch.shape[1], ycorner:] = 1. - batch[:, :xcorner + block_size -
                                                                                                 batch.shape[1],
                                                                                       ycorner:]

                    out_c1[:, xcorner:, :ycorner + block_size - batch.shape[2]] = batch[:, xcorner:,
                                                                                  :ycorner + block_size - batch.shape[
                                                                                      2]]
                    out_c2[:, xcorner:, :ycorner + block_size - batch.shape[2]] = 1. - batch[:, xcorner:,
                                                                                       :ycorner + block_size -
                                                                                        batch.shape[2]]

                    out_c1[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = batch[:,
                                                                                                                :xcorner + block_size -
                                                                                                                 batch.shape[
                                                                                                                     1],
                                                                                                                :ycorner + block_size -
                                                                                                                 batch.shape[
                                                                                                                     2]]
                    out_c2[:, :xcorner + block_size - batch.shape[1],
                    :ycorner + block_size - batch.shape[2]] = 1. - batch[:, :xcorner + block_size - batch.shape[1],
                                                                   :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[:, xcorner:, ycorner:ycorner + block_size] = batch[:, xcorner:, ycorner:ycorner + block_size]
                    out_c2[:, xcorner:, ycorner:ycorner + block_size] = 1. - batch[:, xcorner:,
                                                                             ycorner:ycorner + block_size]

                    out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size] = batch[:,
                                                                                                      :xcorner + block_size -
                                                                                                       batch.shape[1],
                                                                                                      ycorner:ycorner + block_size]
                    out_c2[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size] = 1. - batch[:,
                                                                                                           :xcorner + block_size -
                                                                                                            batch.shape[
                                                                                                                1],
                                                                                                           ycorner:ycorner + block_size]
            else:
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[:, xcorner:xcorner + block_size, ycorner:] = batch[:, xcorner:xcorner + block_size, ycorner:]
                    out_c2[:, xcorner:xcorner + block_size, ycorner:] = 1. - batch[:, xcorner:xcorner + block_size,
                                                                             ycorner:]

                    out_c1[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]] = batch[:,
                                                                                                      xcorner:xcorner + block_size,
                                                                                                      :ycorner + block_size -
                                                                                                       batch.shape[2]]
                    out_c2[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]] = 1. - batch[:,
                                                                                                           xcorner:xcorner + block_size,
                                                                                                           :ycorner + block_size -
                                                                                                            batch.shape[
                                                                                                                2]]
                else:
                    out_c1[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = batch[:,
                                                                                            xcorner:xcorner + block_size,
                                                                                            ycorner:ycorner + block_size]
                    out_c2[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = 1. - batch[:,
                                                                                                 xcorner:xcorner + block_size,
                                                                                                 ycorner:ycorner + block_size]

            out_c1 = out_c1.permute(0, 3, 1, 2)
            out_c2 = out_c2.permute(0, 3, 1, 2)
            out = torch.cat((out_c1, out_c2), 1)
            final_output.append(out)
    return final_output


def generate_smoothing_units(inputs, band_size_dict):
    """
    generate smoothing block/col/row units
    Args:
        inputs: samples
        band_size_dict: we need smoothing size to smoothing

    Returns:smoothing block/col/row units

    """
    blocks = generate_blocks(inputs, band_size_dict.get("block"))
    columns = generate_columns(inputs, band_size_dict.get("column"))
    rows = generate_rows(inputs, band_size_dict.get("row"))
    units_dict = {"block": blocks, "column": columns, "row": rows}
    return units_dict


def generate_prediction_for_units(inputs, units_dict, model_dict, num_classes=10, threshold=0.3):
    """
    func for predition
    Args:
        inputs: sample
        units_dict: smoothing units
        model_dict: model trained by smoothing units
        num_classes: 10 in cifar
        threshold: 0.3 is optimal

    Returns: softmx_dict, vote_sum_dict, vote_each_list_dict

    """
    softmx_dict = {}
    vote_sum_dict = {}
    vote_each_list_dict = {}
    # for every smoothing method
    for model_name in model_dict:
        # for softmx result & vote
        model_softmx_result = []
        model_vote_single = []
        # for the sum of vote from each units
        model_vote_sum = torch.zeros(inputs.size(0), num_classes).type(torch.int).cuda()
        # get the trained smoothing model
        net = model_dict.get(model_name)
        # get the smoothing units from previous ablation
        units = units_dict.get(model_name)
        # for each units
        for unit in units:
            # put into model
            softmx = torch.nn.functional.softmax(net(unit), dim=1)
            # put to list
            model_softmx_result.append(softmx)
            # calculate the sum of vote
            model_vote_single.append((softmx >= threshold).type(torch.int).cuda())
            model_vote_sum += (softmx >= threshold).type(torch.int).cuda()
        # put data into dist
        softmx_dict[model_name] = model_softmx_result
        vote_sum_dict[model_name] = model_vote_sum
        vote_each_list_dict[model_name] = model_vote_single
    return softmx_dict, vote_sum_dict, vote_each_list_dict

def static_certify(predictions, size_to_certify, smoothing_method, band_size_dict):
    risky_label = []
    predinctionsnp = predictions.cpu().numpy()
    # sort the label
    idxsort = np.argsort(-predinctionsnp, axis=1, kind='stable')
    # sort the vote
    valsort = -np.sort(-predinctionsnp, axis=1, kind='stable')
    # pick first
    val = valsort[:, 0]
    idx = idxsort[:, 0]
    # pick second
    valsecond = valsort[:, 1]
    idxsecond = idxsort[:, 1]
    if smoothing_method == "row" or smoothing_method == "column":
        # just one
        num_affected_classifications = (size_to_certify + band_size_dict.get(smoothing_method) - 1)
    else:
        # sqaure
        num_affected_classifications = (size_to_certify + band_size_dict.get(smoothing_method) - 1) * (
                size_to_certify + band_size_dict.get(smoothing_method) - 1)
    # check whether certify
    cert = torch.tensor(((val - valsecond > 2 * num_affected_classifications) | (
            (val - valsecond == 2 * num_affected_classifications) & (idx < idxsecond)))).cuda()
    return torch.tensor(idx).cuda(), torch.tensor(idxsecond).cuda(), val, valsecond, cert


def quality_of_output_wrongcert(idx_first, targets, cert, smoothing_method, clean_correct_dict, cert_correct_dict,
                                cert_incorrect_dict):
    # correct output
    correct_output = idx_first.eq(targets).sum().item()
    # correct and certify output
    correct_and_cert_output = (idx_first.eq(targets) & cert).sum().item()
    incorrect_and_cert_output = (idx_first.eq(targets).logical_not() & cert).sum().item()

    if clean_correct_dict.get(smoothing_method) is None:
        clean_correct_dict[smoothing_method] = correct_output
        cert_correct_dict[smoothing_method] = correct_and_cert_output
        cert_incorrect_dict[smoothing_method] = incorrect_and_cert_output
    else:
        clean_correct_dict[smoothing_method] = clean_correct_dict.get(smoothing_method) + correct_output
        cert_correct_dict[smoothing_method] = cert_correct_dict.get(
            smoothing_method) + correct_and_cert_output
        cert_incorrect_dict[smoothing_method] = cert_incorrect_dict.get(smoothing_method) + incorrect_and_cert_output
    return clean_correct_dict, cert_correct_dict, cert_incorrect_dict


def quality_of_output_wrongcert_theo(idx_first, targets, cert, smoothing_method, clean_correct_dict, cert_correct_dict,
                                cert_incorrect_dict,the1_cert_correct,the2_cert_correct,the1_cert_incorrect,the2_cert_incorrect,theo_num):
    # correct output
    correct_output = idx_first.eq(targets).sum().item()
    # correct and certify output
    correct_and_cert_output = (idx_first.eq(targets) & cert).sum().item()
    incorrect_and_cert_output = (idx_first.eq(targets).logical_not() & cert).sum().item()
    if correct_and_cert_output==1:
        if theo_num==1:
            the1_cert_correct+=1
        if theo_num==2:
            the2_cert_correct += 1
    if incorrect_and_cert_output==1:
        if theo_num==1:
            the1_cert_incorrect+=1
        if theo_num==2:
            the2_cert_incorrect += 1

    if clean_correct_dict.get(smoothing_method) is None:
        clean_correct_dict[smoothing_method] = correct_output
        cert_correct_dict[smoothing_method] = correct_and_cert_output
        cert_incorrect_dict[smoothing_method] = incorrect_and_cert_output
    else:
        clean_correct_dict[smoothing_method] = clean_correct_dict.get(smoothing_method) + correct_output
        cert_correct_dict[smoothing_method] = cert_correct_dict.get(
            smoothing_method) + correct_and_cert_output
        cert_incorrect_dict[smoothing_method] = cert_incorrect_dict.get(smoothing_method) + incorrect_and_cert_output
    return clean_correct_dict, cert_correct_dict, cert_incorrect_dict, the1_cert_correct, the2_cert_correct, the1_cert_incorrect, the2_cert_incorrect

def result_shower_incorrect_cert(static_clean_correct_dict, static_cert_correct_dict, cert_incorrect_dict, total, time,
                                 technique):
    print("using time:" + str(time))
    for smoothing_method in static_clean_correct_dict:
        clean_correct = static_clean_correct_dict.get(smoothing_method)
        cert_correct = static_cert_correct_dict.get(smoothing_method)
        cert_incorrect = cert_incorrect_dict.get(smoothing_method)
        print(technique)
        print(smoothing_method)
        print("clean_correct" + str(clean_correct))
        print("cert_correct" + str(cert_correct))
        print("cert_incorrect" + str(cert_incorrect))
        print("clean_correct_precent" + str(clean_correct / total))
        print("cert_correct_precent" + str(cert_correct / total))
        print("cert_incorrect_precent" + str(cert_incorrect / total))

        print("total" + str(total))
        print("\n")
    print("This turn, result end")


def idx_corrector(idx_list, new_idx):
    if new_idx not in idx_list:
        idx_list.append(new_idx)
    return idx_list


def block_location(x, y, width):
    location = y * width + x
    return location


def single_band_scanner(idx, band_size, malicious_label, each_vote, size_to_certify):
    idx_list = []
    new_each_vote = each_vote.copy()
    """down"""
    for down in range(size_to_certify):
        zero = torch.zeros([1, num_class]).type(torch.int).cuda()
        zero[:, malicious_label] = 1
        new_each_vote[idx + down] = zero
        idx_list = idx_corrector(idx_list, idx + down)
    """up"""
    for up in range(band_size):
        zero = torch.zeros([1, num_class]).type(torch.int).cuda()
        zero[:, malicious_label] = 1
        new_each_vote[idx + up - band_size + 1] = zero
        idx_list = idx_corrector(idx_list, idx + up - band_size + 1)
    if len(idx_list) != band_size + size_to_certify - 1:
        print("Error")
    return new_each_vote


def single_block_scanner(x, y, block_size, malicious_label, each_vote, size_to_certify):
    """a new patch, we assume this patch can attack successfully"""
    new_each_vote = each_vote.copy()
    idx_list = []
    length = int(sqrt(len(new_each_vote)))
    """right"""
    for right in range(size_to_certify):
        for up in range(block_size):
            zero = torch.zeros([1, num_class]).type(torch.int).cuda()
            zero[:, malicious_label] = 1
            if y - up >= 0:
                idx = block_location(x + right, y - up, length)
            else:
                idx = block_location(x + right, length + y - up, length)
            new_each_vote[idx] = zero
            idx_list = idx_corrector(idx_list, idx)
        for down in range(size_to_certify):
            zero = torch.zeros([1, num_class]).type(torch.int).cuda()
            zero[:, malicious_label] = 1
            idx = block_location(x + right, y + down, length)
            new_each_vote[idx] = zero
            idx_list = idx_corrector(idx_list, idx)
    for left in range(block_size):
        for up in range(block_size):
            zero = torch.zeros([1, num_class]).type(torch.int).cuda()
            zero[:, malicious_label] = 1
            if x - left >= 0:
                if y - up >= 0:
                    idx = block_location(x - left, y - up, length)
                else:
                    idx = block_location(x - left, length + y - up, length)
            else:
                if y - up >= 0:
                    idx = block_location(length + x - left, y - up, length)
                else:
                    idx = block_location(length + x - left, length + y - up, length)
            new_each_vote[idx] = zero
            idx_list = idx_corrector(idx_list, idx)
        for down in range(size_to_certify):
            zero = torch.zeros([1, num_class]).type(torch.int).cuda()
            zero[:, malicious_label] = 1
            if x - left >= 0:
                idx = block_location(x - left, y + down, length)
            else:
                idx = block_location(x - left + length, y + down, length)
            new_each_vote[idx] = zero
            idx_list = idx_corrector(idx_list, idx)
    if len(idx_list) != (size_to_certify + block_size - 1) * (size_to_certify + block_size - 1):
        print("wrong!!!!")
    return new_each_vote

# for DRS
def malicious_label_check(predictions, size_to_certify, smoothing_method, band_size_dict):
    malicious_label = []
    predinctionsnp = predictions.cpu().numpy()
    idxsort = np.argsort(-predinctionsnp, axis=1, kind='stable')
    valsort = -np.sort(-predinctionsnp, axis=1, kind='stable')
    val = valsort[:, 0]
    idx = idxsort[:, 0]
    if smoothing_method == "row" or smoothing_method == "column":
        num_affected_classifications = (size_to_certify + band_size_dict.get(smoothing_method) - 1)
    elif smoothing_method == "block":
        num_affected_classifications = (size_to_certify + band_size_dict.get(smoothing_method) - 1) * (
                size_to_certify + band_size_dict.get(smoothing_method) - 1)
    elif smoothing_method == "multi":
        num_affected_classifications = (size_to_certify + band_size_dict.get(smoothing_method) - 1) * 3
    not_first = True
    for i in range(len(idxsort[0])):
        if not_first:
            not_first = False
            continue
        else:
            cert_for_single_label = torch.tensor(((val - valsort[:, i] > 2 * num_affected_classifications) | (
                    (val - valsort[:, i] == 2 * num_affected_classifications) & (idx < idxsort[:, i])))).cuda()
            if cert_for_single_label == True:
                continue
            else:
                malicious_label.append(idxsort[:, i])
    return torch.tensor(idx).cuda(), malicious_label

def trans_softmx_to_eachvote(softmaxs, threshhold):
    vote = []
    for softmax in softmaxs:
        vote.append((softmax >= threshhold).type(torch.int).cuda())
    return vote


def trans_eachvote_to_sumvote(each_vote):
    sum_vote = torch.zeros(len(each_vote[0]), num_class).type(torch.int).cuda()
    for vote in each_vote:
        sum_vote += vote
    return sum_vote


def add_extand_to_sumvote(sum_vote, extand_vote):
    sum_vote = sum_vote + extand_vote
    return sum_vote


def trans_sumvote_to_winner(sum_vote):
    sum_votenp = sum_vote.cpu().numpy()
    idxsort = np.argsort(-sum_votenp, axis=1, kind='stable')
    valsort = -np.sort(-sum_votenp, axis=1, kind='stable')
    val = valsort[:, 0]
    idx = idxsort[:, 0]
    val_second = valsort[:, 1]
    idx_second = idxsort[:, 1]
    return torch.tensor(idx).cuda(), torch.tensor(idx_second).cuda(), val, val_second

def the3_record(predi_label, cert, predi_dict, cert_dict):
    # prediction output
    if predi_dict.get(predi_label) is None:
        predi_dict[predi_label] = 1
    else:
        predi_dict[predi_label] = predi_dict.get(predi_label) + 1

    if cert_dict.get(predi_label) is None:
        if cert:
            cert_dict[predi_label] = 1
    else:
        if cert:
            cert_dict[predi_label] = cert_dict.get(predi_label) + 1

    return predi_dict, cert_dict


# Attention Here the number of classifier is 3
def the3_output(predi_dict, cert_dict):
    prediction_output = -1
    cert_output = False
    for label in predi_dict:
        if predi_dict[label] >= 2:
            prediction_output = label
    for label in cert_dict:
        if label != prediction_output:
            continue
        if cert_dict[label] >= 2:
            cert_output = True
    return prediction_output, cert_output

def position_to_dict(location_malicious_dict, x, y, malicious_label):
    location_malicious_dict[x, y].append(malicious_label)
    return location_malicious_dict

def dynamic_single_certify_forthe4_fast(softmx_dict, size_to_certify, band_size_dict, threshhold):
    each_vote_dict = {}
    malicious_label_list_dict = {}
    predi_dict = {}
    cert_dict = {}
    if_smoothing_method_cert_dict = {}
    smoothing_method_predi_dict = {}
    theo=-1
    for smoothing_method in band_size_dict:
        cert = True
        # location_malicious_dict = defaultdict(list)
        each_vote = trans_softmx_to_eachvote(softmx_dict.get(smoothing_method), threshhold)
        each_vote_dict[smoothing_method] = each_vote
        sum_vote = trans_eachvote_to_sumvote(each_vote)
        old_winner, malicious_label_list = malicious_label_check(sum_vote, size_to_certify, smoothing_method,
                                                                 band_size_dict)
        if len(malicious_label_list) == 0:
            cert = True
        else:
            cert = False
        if_smoothing_method_cert_dict[smoothing_method] = cert
        smoothing_method_predi_dict[smoothing_method] = old_winner
        # # you dont know if the winner is majority or not
        malicious_label_list.append(old_winner)
        # here, malicious_label is for DRS, not the local-malicious label, so we will enable it below
        malicious_label_list_dict[smoothing_method] = malicious_label_list
        predi_dict, cert_dict = the3_record(turn_tensor_to_int(old_winner), cert, predi_dict, cert_dict)
    prediction_output, cert_output = the3_output(predi_dict, cert_dict)
    if prediction_output == -1:
        label_list = []
        "output the smallest one"
        for smoothing_method in smoothing_method_predi_dict:
            label_list.append(smoothing_method_predi_dict.get(smoothing_method))
        prediction_output = the4_winner_judge(label_list)
        prediction_output = turn_tensor_to_int(prediction_output)
    else:
        prediction_output = prediction_output
    if cert_output == True:
        print("Theo1 pass")
        theo=1
        return prediction_output, cert_output,theo
    else:
        theo=2
        print("Theo1 fail")

    # scan for every location
    block_width = int(len(softmx_dict.get('row')))
    for idx_x in range(block_width - size_to_certify):
        for idx_y in range(block_width - size_to_certify):
            new_winner_list_dict = defaultdict(list)
            for smoothing_method in each_vote_dict:
                if if_smoothing_method_cert_dict.get(smoothing_method) == True:
                    pre_cert = smoothing_method_predi_dict.get(smoothing_method)
                    new_winner_list_dict[smoothing_method].append(pre_cert)
                    continue
                band_size = band_size_dict.get(smoothing_method)
                each_vote = each_vote_dict.get(smoothing_method)
                # malicious_label_list = malicious_label_list_dict.get(smoothing_method)
                # check for each one
                malicious_label_list = [0,1,2,3,4,5,6,7,8,9]
                # here is the local-malicious label check
                if smoothing_method == "block":
                    # sample size
                    """a new patch, we assume this patch can attack successfully"""
                    # for malicious_label
                    for malicious_label in malicious_label_list:
                        # scan for a single place
                        new_each_vote = single_block_scanner(idx_x, idx_y, band_size_dict.get(smoothing_method),
                                                             malicious_label, each_vote, size_to_certify)
                        # cal the sum
                        new_sum_vote = trans_eachvote_to_sumvote(new_each_vote)
                        # cal new winner
                        new_winner, new_second, new_winner_val, new_second_val = trans_sumvote_to_winner(new_sum_vote)
                        if new_winner_val == new_second_val and new_second < new_winner:
                            new_winner_list_dict[smoothing_method].append(new_second)
                        else:
                            new_winner_list_dict[smoothing_method].append(new_winner)

                if smoothing_method == "column":
                    """a new patch, we assume this patch can attack successfully"""
                    for malicious_label in malicious_label_list:
                        new_each_vote = single_band_scanner(idx_x, band_size, malicious_label, each_vote,
                                                            size_to_certify)
                        new_sum_vote = trans_eachvote_to_sumvote(new_each_vote)
                        new_winner, new_second, new_winner_val, new_second_val = trans_sumvote_to_winner(new_sum_vote)
                        # need to break the tie
                        if new_winner_val == new_second_val and new_second < new_winner:
                            new_winner_list_dict[smoothing_method].append(new_second)
                        else:
                            new_winner_list_dict[smoothing_method].append(new_winner)

                if smoothing_method == "row":
                    """a new patch, we assume this patch can attack successfully"""
                    for malicious_label in malicious_label_list:
                        new_each_vote = single_band_scanner(idx_y, band_size, malicious_label, each_vote,
                                                            size_to_certify)
                        new_sum_vote = trans_eachvote_to_sumvote(new_each_vote)
                        new_winner, new_second, new_winner_val, new_second_val = trans_sumvote_to_winner(new_sum_vote)
                        if new_winner_val == new_second_val and new_second < new_winner:
                            new_winner_list_dict[smoothing_method].append(new_second)
                        else:
                            new_winner_list_dict[smoothing_method].append(new_winner)

            same_or_not = check_same(new_winner_list_dict, prediction_output)
            if same_or_not:
                continue
            else:
                print("Theo4 fail!!!")
                return prediction_output, cert_output, theo
    cert_output = True
    print("Theo4 success!!!")
    return prediction_output, cert_output, theo

def check_same(new_winner_list_dict, old_winner, total=3):
    same = True
    block_label_list = new_winner_list_dict.get('block')
    column_label_list = new_winner_list_dict.get('column')
    row_label_list = new_winner_list_dict.get('row')
    for block_label in block_label_list:
        for column_label in column_label_list:
            for row_label in row_label_list:
                list = [block_label, column_label, row_label]
                tmp_result = the4_winner_judge(list)
                if tmp_result != old_winner:
                    same = False
                    return same
    return same



def the4_output(prediction_output, cert_output, single_result_record_dict):
    if cert_output == True:
        return prediction_output, cert_output
    else:
        if prediction_output == -1:
            label_list = []
            "output the smallest one"
            for smoothing_method in single_result_record_dict:
                label_list.append(single_result_record_dict.get(smoothing_method))
            previous_winner = the4_winner_judge(label_list)
        else:
            previous_winner = prediction_output

        return previous_winner, cert_output


# attention
def the4_winner_judge(label_list):
    label0 = label_list[0]
    label1 = label_list[1]
    label2 = label_list[2]

    if label0 == label1 or label0 == label2:
        return label0
    elif label1 == label2:
        return label1
    else:
        if label0 < label2 < label1 or label0 < label1 < label2:
            return label0
        elif label1 < label0 < label2 or label1 < label2 < label0:
            return label1
        else:
            return label2

def turn_int_to_tensor(int_value):
    a = np.array(int_value)
    tensor_out = torch.from_numpy(a)
    return tensor_out


def turn_tensor_to_int(tensor_value):
    idx_first = tensor_value.data.cpu().numpy()[0]
    return idx_first



