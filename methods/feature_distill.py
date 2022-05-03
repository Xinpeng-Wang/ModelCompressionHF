def att_mse_hidden_mse(student_att, teacher_att, student_rep, teacher_rep):
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

        tmp_loss = loss_mse(student_att, teacher_att)
        att_loss += tmp_loss

    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
    new_student_reps = student_reps
    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
        tmp_loss = loss_mse(student_rep, teacher_rep)
        rep_loss += tmp_loss
    

    return rep_loss, att_loss 




def att_kl(student_att, teacher_att):
    att_loss = 0.
    rep_loss = 0.


    teacher_layer_num = len(teacher_atts)
    student_layer_num = len(student_atts)
    assert teacher_layer_num % student_layer_num == 0
    layers_per_block = int(teacher_layer_num / student_layer_num)
    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                        for i in range(student_layer_num)]

    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (student_att.shape[0] * student_att.shape[1]) #, reduction='batchmean', log_target=True)

        att_loss += loss_kl_tmp


    return rep_loss, att_loss


    