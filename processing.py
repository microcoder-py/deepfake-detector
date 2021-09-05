def processing(location = "", save_path = "", capture_sec = 5, num_vids = 64):

    import json
    import shutil
    num_frames = int(capture_sec*60)

    metadata = json.load(open(location+"/metadata.json"))
    shutil.copy2(location+"/metadata.json", save_path)

    count_fake = 0
    count_real = 0

    keys = metadata.keys()

    list_real = []
    list_fake = []
    list_labels = []

    for key in keys:
        if(metadata[key]["label"] == "FAKE"):
            list_fake.append(location + "/" + key)
        else:
            list_real.append(location + "/" + key )

    list_fake = list_fake[0:num_vids]
    list_real = list_real[0:num_vids]

    fin_list = list_fake + list_real

    import os
    import glob
    file_set = glob.glob(location+"/*.mp4")

    for i in range(len(file_set)):
        file_set[i] = file_set[i].replace("\\", "/")

    for i in range(len(file_set)):
        if(file_set[i] not in fin_list):
            os.remove(file_set[i])

    import generate_preprocessed_files as preprocessing

    preprocessing.generate(path_root = location, save_path = save_path, capture_sec = capture_sec)
