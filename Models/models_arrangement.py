import numba as nb
import numpy as np


@nb.njit
def error_calculator(
    number_of_mobile_devices__k: int, 
    data_dimension__L: int, 
    number_of_parallel_channels__M: int,
    probability_that_user_can_compute_its_local_update__pcomp: float, 
    number_of_iterations__t: int,
    learning_rate__u1: float, 
    step_size__u: float,
    clusters_list: list
):
    ###########################################################################
    # Initialization of inputs and outputs.
    users_input__x = np.random.randn(number_of_mobile_devices__k, data_dimension__L) # "x𝑘" from each device.
    weights_vector__w = np.random.randn(data_dimension__L, ) # "w".
    users_output__y = np.dot(users_input__x, weights_vector__w) # "y𝑘" from each device.

    ###########################################################################
    # SGD random start declaration.
    SGD_random_weight_vector = np.random.randn(data_dimension__L, )

    # It is created copies of the SGD in order to make the scenarios a bit more fairer, 
    # altough in practice it would not make much difference if it was randomly sorted 
    # again these values.
    SGD_weight_vector_of_model_1 = np.copy(SGD_random_weight_vector)
    SGD_weight_vector_of_model_2 = np.copy(SGD_random_weight_vector)
    SGD_weight_vector_of_model_3 = np.copy(SGD_random_weight_vector)
    SGD_weight_vector_of_model_1_with_d2d = np.copy(SGD_random_weight_vector)
    SGD_weight_vector_of_model_2_with_d2d = np.copy(SGD_random_weight_vector)
    SGD_weight_vector_of_model_3_with_d2d = np.copy(SGD_random_weight_vector)

    ###########################################################################
    # Variables for the D2D clustering arrangement.
    number_of_clusterheads = len(clusters_list)
    list_of_clusterheads = [cluster[0] for cluster in clusters_list] # Check which index to use in the "clusters header" inside the "proposed_clustering_algorithm.py" module.

    # Calculate the clustering rate.
    number_of_unclustered_devices = 0
    for cluster in clusters_list:
        if len(cluster) == 1:
            number_of_unclustered_devices += 1
    clusterized_devices_rate = (1 - number_of_unclustered_devices/number_of_mobile_devices__k)*100

    # Clusterheads probability variable initialization.
    clusterheads_probability = np.zeros((number_of_clusterheads, ))

    ###########################################################################
    # Variables accessed by model 1 (Polling) alone.
    # Standard arrangement.
    quantity_of_uploads_in_FL_round = np.arange(1, (number_of_iterations__t * number_of_parallel_channels__M) + 1) # Total quantity for this FL round.
    # The below line arranges the users in a way that dictates who will be the next to participate, in accordance with polling strategy.
    order_of_users_that_will_send_data = np.mod(quantity_of_uploads_in_FL_round - 1, number_of_mobile_devices__k)

    # D2D clustering arrangement.
    order_of_clusterheads_that_will_send_data = list_of_clusterheads.copy()
    while len(order_of_clusterheads_that_will_send_data) < (number_of_iterations__t*number_of_parallel_channels__M):
        order_of_clusterheads_that_will_send_data = order_of_clusterheads_that_will_send_data + list_of_clusterheads
    
    ###########################################################################
    # Local updates declaration.
    # Standard arrangement.
    local_updates_model_1_wk = np.zeros((number_of_mobile_devices__k, data_dimension__L)) # Declaration to increase numpy performance.
    local_updates_model_2_wk = np.zeros((number_of_mobile_devices__k, data_dimension__L))
    local_updates_model_3_wk = np.zeros((number_of_mobile_devices__k, data_dimension__L))
    norm_of_local_updates_of_model_3__wi = np.zeros((number_of_mobile_devices__k, ))

    # D2D clustering arrangement.
    local_updates_model_1_with_d2d__wk = np.zeros((number_of_mobile_devices__k, data_dimension__L))
    local_updates_model_2_with_d2d__wk = np.zeros((number_of_mobile_devices__k, data_dimension__L))
    local_updates_model_3_with_d2d__wk = np.zeros((number_of_mobile_devices__k, data_dimension__L))
    # Note: now, the norm of the local updates that should be done involves 
    # fisrt aggregating the local updates of the cluster members into the clusterheads, 
    # and only then taking the norm.
    model_3_with_d2d_local_updates_aggregated_in_the_clusterheads = np.zeros((number_of_clusterheads, data_dimension__L))
    norm_of_aggregated_updates_of_model_3_with_d2d__wi = np.zeros((number_of_clusterheads, ))

    ###########################################################################
    # Variable accessed by models 2 (ALOHA with fixed access probability) 
    # and 3 (ALOHA with optimized access probability).
    # Standard arrangement.
    access_probability = number_of_parallel_channels__M / number_of_mobile_devices__k # Reference: second line after formula 4.
    
    # D2D clustering arrangement.
    access_probability_d2d = number_of_parallel_channels__M / number_of_clusterheads

    ###########################################################################
    # Variable accessed by model 3 (ALOHA with optimized access probability) alone.
    # Standard arrangement.
    psi = 0 # In document: ψ. This is the feedback signal from the BS at the end of each 
    # FL iteration. It is declared with the value 0, but it will be updated accordingly to formula 12.
    
    # D2D clustering arrangement.
    psi_d2d = 0

    ###########################################################################
    # Variables to check the number of uploads.
    # Standard arrangement.
    model_1_total_of_successful_uploads = 0
    model_2_total_of_successful_uploads = 0
    model_3_total_of_successful_uploads = 0

    # D2D clustering arrangement.
    model_1_d2d_total_of_successful_uploads = 0
    model_2_d2d_total_of_successful_uploads = 0
    model_3_d2d_total_of_successful_uploads = 0

    model_1_d2d_total_of_successful_clusterheads_uploads = 0
    model_2_d2d_total_of_successful_clusterheads_uploads = 0
    model_3_d2d_total_of_successful_clusterheads_uploads = 0

    ###########################################################################
    # Start of the iterations of the FL round. This 'for loop' stands for 
    # a single FL round, compose of 't' iterations.
    for iteration_t in range(number_of_iterations__t):
        # Defining the probabilities.
        devices_probability = np.random.rand(number_of_mobile_devices__k, )
        channels_probability = np.random.rand(number_of_parallel_channels__M, )

        for clusterhead_index, clusterhead in enumerate(list_of_clusterheads):
            clusterheads_probability[clusterhead_index] = devices_probability[clusterhead]

        ###########################################################################
        # Error of the models. Reference: based on Choi's article, this is the implementation
        # of formula 15.
        # In the adopted strategy, the sum of the local updates of the devices that
        # transmitted successfully is used as the gradient in the SGD step.
        # This phase is considered as the update of the models accordingly to the 
        # latest model that the BS is broadcasting.
        err_model_1 = (users_input__x @ SGD_weight_vector_of_model_1) - users_output__y
        err_model_1_d2d = (users_input__x @ SGD_weight_vector_of_model_1_with_d2d) - users_output__y

        err_model_2 = (users_input__x @ SGD_weight_vector_of_model_2) - users_output__y 
        err_model_2_d2d = (users_input__x @ SGD_weight_vector_of_model_2_with_d2d) - users_output__y

        err_model_3 = (users_input__x @ SGD_weight_vector_of_model_3) - users_output__y
        err_model_3_d2d = (users_input__x @ SGD_weight_vector_of_model_3_with_d2d) - users_output__y

        for counter in range(number_of_mobile_devices__k):
            # Reference: the local updates goes accordingly to formula 7, or 2 from Choi.
            local_updates_model_1_wk[counter, :] = err_model_1[counter] * users_input__x[counter, :]
            local_updates_model_1_with_d2d__wk[counter, :] = err_model_1_d2d[counter] * users_input__x[counter, :]

            local_updates_model_2_wk[counter, :] = err_model_2[counter] * users_input__x[counter, :]
            local_updates_model_2_with_d2d__wk[counter, :] = err_model_2_d2d[counter] * users_input__x[counter, :]

            local_updates_model_3_wk[counter, :] = err_model_3[counter] * users_input__x[counter, :]
            local_updates_model_3_with_d2d__wk[counter, :] = err_model_3_d2d[counter] * users_input__x[counter, :]

            # Model 3 takes the norm of the local updates into consideration.
            # Note: Unfortunetely, numba does not accept the np.linalg.norm function 
            # with the axis param, so it is not possible to call a one line code like: 
            # "norm_of_local_updates_of_model_3__wi = np.linalg.norm(local_updates_model_3_wk, axis=1)".
            local_update_model_3_wk = err_model_3[counter] * users_input__x[counter]
            norm_of_local_updates_of_model_3__wi[counter] = np.linalg.norm(local_update_model_3_wk)

            # Note: for the model 3 with D2D clustering, it is needed to aggregate in the 
            # clusterhead the local updates of the cluster members first, and only then
            # taking the norm of clusterhead. Therefore, it is not updated for all devices, but
            # instead only for the clusterheads properly.
        # The below code implements the norm of the aggregated clusterheads, or the hierarchical clustering.
        for idx, cluster in enumerate(clusters_list):
            cluster_as_array = np.array(cluster) # Done for running numba.
            if len(cluster)>= 2:
                model_3_with_d2d_local_updates_aggregated_in_the_clusterheads[idx] = np.sum(local_updates_model_3_with_d2d__wk[cluster_as_array, :], axis=0)
            else:
                model_3_with_d2d_local_updates_aggregated_in_the_clusterheads[idx] = local_updates_model_3_with_d2d__wk[cluster_as_array[0]] # Only the information of the clusterhead.
            # Like explained before, instead of taking the norm of the local updates
            # of each user, it is taken the norm of the aggregated updates of the clusterheads.
            norm_of_aggregated_updates_of_model_3_with_d2d__wi[idx] = np.linalg.norm(model_3_with_d2d_local_updates_aggregated_in_the_clusterheads[idx])

        ###########################################################################
        # MODEL 1: Pooling.
        # In this strategy, the Base Station (BS) asks for some batch of users
        # to upload their local updates.

        batch_of_users_asked_to_upload = order_of_users_that_will_send_data[
            iteration_t * number_of_parallel_channels__M: (iteration_t+1) * number_of_parallel_channels__M]
        # The batch works this way: if we have 10 channels, then at iteration 1
        # the users 1:10 are chosen as the batch. Then, at iteration 2, users 11:20 
        # are the batch, and so on... If we have more iterations to be done than we
        # have of users, we start all over again at user 1.
        indexes_where_channels_probability_are_lesser_than_pcomp = np.where(
            channels_probability < probability_that_user_can_compute_its_local_update__pcomp)[0]
        
        users_asked_to_upload_that_are_truly_able_to_send = batch_of_users_asked_to_upload[
            indexes_where_channels_probability_are_lesser_than_pcomp].astype(np.int32)

        model_1_total_of_successful_uploads += len(users_asked_to_upload_that_are_truly_able_to_send)

        gradient_of_model_1 = np.sum(
            local_updates_model_1_wk[users_asked_to_upload_that_are_truly_able_to_send, :], axis=0)
        SGD_weight_vector_of_model_1 = SGD_weight_vector_of_model_1 - np.multiply(learning_rate__u1, gradient_of_model_1)

        ###########################################################################
        # MODEL 1 with D2D clustering.
        # This time, instead of asking for a single device to upload its local updates,
        # it is asked for a clusterhead (and consequently all the members that belong
        # in the same cluster of that clusterhead, which will send their data through the
        # clusterhead) to upload its local updates.

        batch_of_clusterheads_asked_to_upload = order_of_clusterheads_that_will_send_data[
            iteration_t * number_of_parallel_channels__M: (iteration_t+1) * number_of_parallel_channels__M]
        batch_of_clusterheads_asked_to_upload = np.array(batch_of_clusterheads_asked_to_upload)

        clusterheads_asked_to_upload_that_are_truly_able_to_send = batch_of_clusterheads_asked_to_upload[
            indexes_where_channels_probability_are_lesser_than_pcomp]

        model_1_d2d_total_of_successful_clusterheads_uploads += len(clusterheads_asked_to_upload_that_are_truly_able_to_send)

        # The below code block implements the hierarchical clustering.
        # Part for calculating the number of transmissions.
        users_asked_to_upload_that_are_truly_able_to_send = [-1] # Numba needs some declaration.
        # Part for calculating the error update at the BS.
        number_of_clusterheads_that_transmitted_successfully = len(clusterheads_asked_to_upload_that_are_truly_able_to_send)
        gradient_of_the_clusterheads = np.zeros((number_of_clusterheads_that_transmitted_successfully, data_dimension__L)) # Variable initialization.
        for idx, clusterhead in enumerate(clusterheads_asked_to_upload_that_are_truly_able_to_send):
            # The below loop seaches for the clusterhead in the list of clusters, 
            # in order to grab its cluster members.
            for cluster in clusters_list:
                if clusterhead in cluster:
                    # Part for calculating the number of transmissions.
                    users_asked_to_upload_that_are_truly_able_to_send = users_asked_to_upload_that_are_truly_able_to_send + cluster
                    # Part for calculating the error update at the BS.
                    cluster_as_array = np.array(cluster) # Numba needs this conversion from python list to numpy array in order to use the np.sum.
                    gradient_of_the_clusterhead = np.sum(local_updates_model_1_with_d2d__wk[cluster_as_array, :], axis=0)
                    gradient_of_the_clusterheads[idx] = gradient_of_the_clusterhead
                    break
                else:
                    pass
        # Part for calculating the number of transmissions.
        users_asked_to_upload_that_are_truly_able_to_send.pop(0) # Removes the "-1" from the declaration of the list because of Numba.
        users_asked_to_upload_that_are_truly_able_to_send = np.array(users_asked_to_upload_that_are_truly_able_to_send).astype(np.int32) # Used because of Numba
        model_1_d2d_total_of_successful_uploads += len(users_asked_to_upload_that_are_truly_able_to_send)
        # Part for calculating the error update at the BS.
        gradient_of_model_1_d2d = np.sum(gradient_of_the_clusterheads[:], axis=0)
        SGD_weight_vector_of_model_1_with_d2d = SGD_weight_vector_of_model_1_with_d2d - np.multiply(learning_rate__u1, gradient_of_model_1_d2d)

        ###########################################################################
        # MODEL 2: Random access without taking into account the norm of the local updates.
        # Check which devices can compute local updates based on probability thresholds.
        indexes_where_devices_probability_are_lesser_than_threshold = np.where(
            devices_probability < min(access_probability, probability_that_user_can_compute_its_local_update__pcomp))[0]

        # Reference: the variable below is Pt at formula 12. This variable can be interpreted as 
        # the number of users that can compute its local updates.
        estimate_of_P_at_iteration_t__Pt = len(indexes_where_devices_probability_are_lesser_than_threshold)

        # Each user may choose (randomly) its channel where it would like to transmit, so
        # the below variable "channel_selected_by_each_user" represents the 
        # channel chosen by each user.
        channel_selected_by_each_user = np.random.randint(1, number_of_parallel_channels__M + 1, estimate_of_P_at_iteration_t__Pt)

        number_of_times_each_channel_was_selected = np.histogram(channel_selected_by_each_user, bins=np.arange(1, number_of_parallel_channels__M + 2))[0]
        # When more than 1 user selects the same channel, a collision happens,
        # and none of the users transmit their uploads. The below variable
        # checks for these collisions conditions.
        # Important note: in Choi's article, he does not consider the
        # scenario where we try to retransmit the information in cases of
        # collision, which is what happens in the real world (soft collision),
        # in accordance with ALOHA protocol.
        
        # Find channels that transmitted without collision.
        channels_that_transmitted_without_collision = np.where(number_of_times_each_channel_was_selected == 1)[0] + 1
        number_of_successful_transmissions = len(channels_that_transmitted_without_collision)
        
        users_that_transmitted_successfully = np.zeros((number_of_successful_transmissions,), dtype=np.uint8)

        for counter in range(number_of_successful_transmissions):
            users_that_transmitted_successfully[counter] = np.where(channel_selected_by_each_user == channels_that_transmitted_without_collision[counter])[0][0]

        model_2_total_of_successful_uploads += len(users_that_transmitted_successfully)

        if len(users_that_transmitted_successfully) >= 1:
            indexes_of_the_users_that_transmitted_sucessfully = indexes_where_devices_probability_are_lesser_than_threshold[users_that_transmitted_successfully]
            gradient_of_model_2 = np.sum(
                local_updates_model_2_wk[indexes_of_the_users_that_transmitted_sucessfully, :], axis=0)
            SGD_weight_vector_of_model_2 = SGD_weight_vector_of_model_2 - np.multiply(learning_rate__u1, gradient_of_model_2)

        ###########################################################################
        # MODEL 2 with D2D clustering.
        # In this arrangement, instead of checking where the probabilities are 
        # lesser than the threshold, so that the devices can send its local updates,
        # it is checked for the clusterheads where their probabilities are lesser
        # than the threshold, so that when an upload is done, it is more significative
        # because it is sending not only the upload of the clusterhead, but all of the
        # cluster members as well.
        # Because we have less communicating devices with the BS (only the clusterheads
        # instead of all of the existing devices), the access probability is changed.
        # The access probability is indirectly changed by the usage of the 
        # "clusterheads_probability".
        indexes_where_clusterheads_probability_are_lesser_than_threshold = np.where(
            clusterheads_probability < min(access_probability_d2d, probability_that_user_can_compute_its_local_update__pcomp))[0]
        
        estimate_of_P_at_iteration_t__Pt = len(indexes_where_clusterheads_probability_are_lesser_than_threshold)

        channel_selected_by_each_clusterhead = np.random.randint(1, number_of_parallel_channels__M + 1, estimate_of_P_at_iteration_t__Pt)
        number_of_times_each_channel_was_selected = np.histogram(channel_selected_by_each_clusterhead, bins=np.arange(1, number_of_parallel_channels__M + 2))[0]
        
        channels_that_transmitted_without_collision = np.where(number_of_times_each_channel_was_selected == 1)[0] + 1
        number_of_successful_transmissions = len(channels_that_transmitted_without_collision)
        
        clusterheads_that_transmitted_successfully = np.zeros((number_of_successful_transmissions, ), dtype=np.uint8)
        for counter in range(number_of_successful_transmissions):
            clusterheads_that_transmitted_successfully[counter] = np.where(channel_selected_by_each_clusterhead == channels_that_transmitted_without_collision[counter])[0][0]

        model_2_d2d_total_of_successful_clusterheads_uploads += len(clusterheads_that_transmitted_successfully)

        # The below code implements the hierarchical clustering.
        number_of_clusterheads_that_transmitted_successfully = len(clusterheads_that_transmitted_successfully)
        gradient_of_the_clusterheads = np.zeros((number_of_clusterheads_that_transmitted_successfully, data_dimension__L)) # Variable initialization.
        for idx, clusterhead in enumerate(clusterheads_that_transmitted_successfully):
            clusterhead_index = indexes_where_clusterheads_probability_are_lesser_than_threshold[clusterhead]
            cluster = clusters_list[clusterhead_index]
            cluster_as_array = np.array(cluster) # Numba needs this conversion from python list to numpy array in order to use the np.sum.
            gradient_of_the_clusterheads[idx] = np.sum(local_updates_model_2_with_d2d__wk[cluster_as_array, :], axis=0)

            model_2_d2d_total_of_successful_uploads += len(cluster)

        gradient_of_model_2_d2d = np.sum(gradient_of_the_clusterheads[:], axis=0)
        SGD_weight_vector_of_model_2_with_d2d = SGD_weight_vector_of_model_2_with_d2d - np.multiply(learning_rate__u1, gradient_of_model_2_d2d)

        ###########################################################################
        # MODEL 3: Random access based averaging.
        # This takes use of the norm of the local updates.

        users_that_can_compute = devices_probability >= probability_that_user_can_compute_its_local_update__pcomp
        # After formula 12 in Choi's article, it is said that if a user cannot compute its local update, 
        # it needs to set ak = 0 so that pk = 0, so this is what the line below is doing. 
        # The name of the variable pk was adapted as pi in the master thesis document, and ak became wi, 
        # which can be seen in formula 11.
        norm_of_local_updates_of_model_3__wi[users_that_can_compute] = 0

        if iteration_t == 0:
            prob_that_BS_receives_local_update_from_user_k__pi = access_probability * np.ones((number_of_mobile_devices__k, )) # Pi* from formula 11.
        else:
            prob_that_BS_receives_local_update_from_user_k__pi = (np.exp(1) * np.log(norm_of_local_updates_of_model_3__wi) - psi)  # Formula 11.
            prob_that_BS_receives_local_update_from_user_k__pi = np.maximum(0, np.minimum(1, prob_that_BS_receives_local_update_from_user_k__pi))

        # Filter devices that can transmit based on probability thresholds.
        indexes_where_devices_probability_are_lesser_than_threshold = np.where(devices_probability < prob_that_BS_receives_local_update_from_user_k__pi)[0]
        estimate_of_P_at_iteration_t__Pt = len(indexes_where_devices_probability_are_lesser_than_threshold)

        # Randomly select a channel for each user that can transmit.
        channel_selected_by_each_user = np.random.randint(1, number_of_parallel_channels__M + 1, estimate_of_P_at_iteration_t__Pt)

        number_of_times_each_channel_was_selected = np.histogram(channel_selected_by_each_user, bins=np.arange(1, number_of_parallel_channels__M + 2))[0]

        # Find channels that transmitted without collision.
        channels_that_transmitted_without_collision = np.where(number_of_times_each_channel_was_selected == 1)[0] + 1
        number_of_successful_transmissions = len(channels_that_transmitted_without_collision)

        psi = psi + step_size__u * (estimate_of_P_at_iteration_t__Pt - number_of_parallel_channels__M)  # Formula 12

        users_that_transmitted_successfully = np.zeros((number_of_successful_transmissions,), dtype=np.uint8)

        for counter in range(number_of_successful_transmissions):
            users_that_transmitted_successfully[counter] = np.where(channel_selected_by_each_user == channels_that_transmitted_without_collision[counter])[0][0]

        model_3_total_of_successful_uploads += len(users_that_transmitted_successfully)

        if len(users_that_transmitted_successfully) >= 1:
            indexes_of_the_users_that_transmitted_sucessfully = indexes_where_devices_probability_are_lesser_than_threshold[users_that_transmitted_successfully]
            gradient_of_model_3 = np.sum(
                local_updates_model_3_wk[indexes_of_the_users_that_transmitted_sucessfully, :], axis=0)
            SGD_weight_vector_of_model_3 = SGD_weight_vector_of_model_3 - np.multiply(learning_rate__u1, gradient_of_model_3)

        ###########################################################################
        # MODEL 3 with d2d clustering.

        # Check for users that can compute. This code below implements the same thing
        # as did the variable "users_that_can_compute" and "model_3_norm_of_local_updates_ak[users_that_can_compute] = 0" 
        # in the original model 3, but this time, as I need to implement some logics involvig the clusterheads
        # and its cluster members, it was easier to implement inside a for loop.
        for clusterhead_index, clusterhead_probability in enumerate(clusterheads_probability):
            if clusterhead_probability >= probability_that_user_can_compute_its_local_update__pcomp:
                # The clusterhead, with the information aggregated from all of its cluster members, 
                # can now compute its local updates.
                norm_of_aggregated_updates_of_model_3_with_d2d__wi[clusterhead_index] = 0

        if iteration_t == 0:
            prob_that_BS_receives_local_update_from_clusterhead_k_pi_d2d = access_probability_d2d * np.ones((number_of_clusterheads, )) # Pi from formula 12.
        else:
            prob_that_BS_receives_local_update_from_clusterhead_k_pi_d2d = np.exp(1)*np.log(norm_of_aggregated_updates_of_model_3_with_d2d__wi) - psi_d2d  # Formula 12
            prob_that_BS_receives_local_update_from_clusterhead_k_pi_d2d = np.maximum(0, np.minimum(1, prob_that_BS_receives_local_update_from_clusterhead_k_pi_d2d))

        # Filter devices that can transmit based on probability thresholds
        indexes_where_clusterheads_probability_are_lesser_than_threshold = np.where(clusterheads_probability < prob_that_BS_receives_local_update_from_clusterhead_k_pi_d2d)[0]
        estimate_of_P_at_iteration_t__Pt = len(indexes_where_clusterheads_probability_are_lesser_than_threshold)

        # Randomly select a channel for each user that can transmit
        channel_selected_by_each_clusterhead = np.random.randint(1, number_of_parallel_channels__M + 1, estimate_of_P_at_iteration_t__Pt)
        number_of_times_each_channel_was_selected = np.histogram(channel_selected_by_each_clusterhead, bins=np.arange(1, number_of_parallel_channels__M + 2))[0]

        # Find channels that transmitted without collision
        channels_that_transmitted_without_collision = np.where(number_of_times_each_channel_was_selected == 1)[0] + 1
        number_of_successful_transmissions = len(channels_that_transmitted_without_collision)

        psi_d2d = psi_d2d + step_size__u * (estimate_of_P_at_iteration_t__Pt - number_of_parallel_channels__M)  # Formula 13

        clusterheads_that_transmitted_successfully = np.zeros((number_of_successful_transmissions,), dtype=np.uint8)
        for counter in range(number_of_successful_transmissions):
            clusterheads_that_transmitted_successfully[counter] = np.where(channel_selected_by_each_clusterhead == channels_that_transmitted_without_collision[counter])[0][0]

        model_3_d2d_total_of_successful_clusterheads_uploads += len(clusterheads_that_transmitted_successfully)

        number_of_clusterheads_that_transmitted_successfully = len(clusterheads_that_transmitted_successfully)
        gradient_of_the_clusterheads = np.zeros((number_of_clusterheads_that_transmitted_successfully, data_dimension__L)) # Variable initialization.
        for idx, clusterhead in enumerate(clusterheads_that_transmitted_successfully):
            clusterhead_index = indexes_where_clusterheads_probability_are_lesser_than_threshold[clusterhead]
            cluster = clusters_list[clusterhead_index]
            cluster_as_array = np.array(cluster) # Numba needs this conversion from python list to numpy array in order to use the np.sum.
            gradient_of_the_clusterheads[idx] = np.sum(local_updates_model_3_with_d2d__wk[cluster_as_array, :], axis=0)

            model_3_d2d_total_of_successful_uploads += len(cluster)
        gradient_of_model_3_d2d = np.sum(gradient_of_the_clusterheads[:], axis=0)
        SGD_weight_vector_of_model_3_with_d2d = SGD_weight_vector_of_model_3_with_d2d - np.multiply(learning_rate__u1, gradient_of_model_3_d2d)

        ###########################################################################
    # Final error calculation.
    error_norm_of_model_1 = np.linalg.norm(SGD_weight_vector_of_model_1 - weights_vector__w)
    error_norm_of_model_1_d2d = np.linalg.norm(SGD_weight_vector_of_model_1_with_d2d - weights_vector__w)
    error_norm_of_model_2 = np.linalg.norm(SGD_weight_vector_of_model_2 - weights_vector__w)
    error_norm_of_model_2_d2d = np.linalg.norm(SGD_weight_vector_of_model_2_with_d2d - weights_vector__w)
    error_norm_of_model_3 = np.linalg.norm(SGD_weight_vector_of_model_3 - weights_vector__w)
    error_norm_of_model_3_d2d = np.linalg.norm(SGD_weight_vector_of_model_3_with_d2d - weights_vector__w)

    return clusterized_devices_rate, \
            error_norm_of_model_1, \
            error_norm_of_model_2, \
            error_norm_of_model_3, \
            error_norm_of_model_1_d2d, \
            error_norm_of_model_2_d2d, \
            error_norm_of_model_3_d2d, \
            model_1_total_of_successful_uploads, \
            model_2_total_of_successful_uploads, \
            model_3_total_of_successful_uploads, \
            model_1_d2d_total_of_successful_uploads, \
            model_2_d2d_total_of_successful_uploads, \
            model_3_d2d_total_of_successful_uploads, \
            model_1_d2d_total_of_successful_clusterheads_uploads, \
            model_2_d2d_total_of_successful_clusterheads_uploads, \
            model_3_d2d_total_of_successful_clusterheads_uploads