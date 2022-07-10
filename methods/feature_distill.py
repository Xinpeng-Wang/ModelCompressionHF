import torch.nn.functional as F
import torch
import math







def att_mse_hidden_mse(student_atts, teacher_atts, student_reps, teacher_reps, device):
    att_loss = 0.
    rep_loss = 0.


    teacher_layer_num = len(teacher_atts)
    student_layer_num = len(student_atts)
    assert teacher_layer_num % student_layer_num == 0
    layers_per_block = int(teacher_layer_num / student_layer_num)
    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                        for i in range(student_layer_num)]

    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                    student_att)
        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                    teacher_att)

        tmp_loss = F.mse_loss(student_att, teacher_att)
        att_loss += tmp_loss

    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
    new_student_reps = student_reps
    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
        tmp_loss = F.mse_loss(student_rep, teacher_rep)
        rep_loss += tmp_loss
    

    return rep_loss, att_loss 




def att_kl(student_atts, teacher_atts):
    att_loss = 0.
    rep_loss = torch.tensor(0.)


    teacher_layer_num = len(teacher_atts)
    student_layer_num = len(student_atts)
    assert teacher_layer_num % student_layer_num == 0
    # 12 / 4 = 3
    # 6 / 4 = 
    # 0 1 2 3 4 5
    layers_per_block = int(teacher_layer_num / student_layer_num)
    # 0123 [2, 5, 8, 11]
    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                        for i in range(student_layer_num)]

    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (student_att.shape[0] * student_att.shape[1]) #, reduction='batchmean', log_target=True)

        att_loss += loss_kl_tmp

    
    return rep_loss, att_loss


def att_kl_4_from_6(student_atts, teacher_atts, layer_selection):
    att_loss = 0.
    rep_loss = torch.tensor(0.)


    teacher_layer_num = len(teacher_atts)
    student_layer_num = len(student_atts)
    # assert teacher_layer_num % student_layer_num == 0
    # 12 / 4 = 3
    # 6 / 4 = 
    # 0 1 2 3 4 5
    # layers_per_block = int(teacher_layer_num / student_layer_num)
    # # 0123 [2, 5, 8, 11]
    # new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
    #                     for i in range(student_layer_num)]
    
    new_teacher_atts = [teacher_atts[i] for i in layer_selection]


    # new_teacher_atts = 

    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (student_att.shape[0] * student_att.shape[1]) #, reduction='batchmean', log_target=True)

        att_loss += loss_kl_tmp

    
    return rep_loss, att_loss


    

def att_val_kl(student_atts_val, teacher_atts_val, device, layer_selection):
    attn_student = [attn[0] for attn in student_atts_val]
            # len(attn_teacher) = 12
    attn_teacher = [attn[0] for attn in teacher_atts_val]



    batch_head_size, length, dk = student_atts_val[0][1].shape
    dk_sqrt = math.sqrt(dk)

    teacher_layer_num = len(attn_teacher)
    student_layer_num = len(attn_student)
    # layers_per_block = 4
    layers_per_block = int(teacher_layer_num / student_layer_num)

    # 能整除的情况下 

    # new_teacher_atts = [attn_teacher[i * layers_per_block + layers_per_block - 1]
    #                             for i in range(student_layer_num)]
    new_teacher_atts = [attn_teacher[i] for i in layer_selection]

    loss_att = 0.
    loss_value = 0.

    for student_att, teacher_att in zip(attn_student, new_teacher_atts):
        # student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
        #                               student_att)
        # teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
        #                               teacher_att)
        student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (batch_head_size * length) #, reduction='batchmean', log_target=True)
        loss_att += loss_kl_tmp

    
    value_student = [attn[1] for attn in student_atts_val]
    value_teacher = [attn[1] for attn in teacher_atts_val]

    
    new_teacher_value = [value_teacher[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]


    for student_value, teacher_value in zip(value_student, new_teacher_value):
        vr_student = F.log_softmax(torch.bmm(student_value, student_value.transpose(1,2))/dk_sqrt, dim=-1)
        vr_teacher = F.softmax(torch.bmm(teacher_value, teacher_value.transpose(1,2))/dk_sqrt, dim=-1)

        loss_value_tmp = F.kl_div(vr_student, vr_teacher, reduction='sum')/(batch_head_size * length)
        loss_value += loss_value_tmp

    
    # loss  = loss_att + loss_value
    return loss_att, loss_value