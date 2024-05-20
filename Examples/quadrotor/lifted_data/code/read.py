# 1. saving trajectories as npz files

import os
import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

import matplotlib.pyplot as plt
import numpy as np

def find_db3_files(directory):
    db3_files = []
    db3_dirs = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            db3_dirs.append(dir)
        for file in files:
            if (file.endswith(".db3")):
                full_path = os.path.join(root, file)
                db3_files.append(full_path)
    return db3_files, db3_dirs

class BagFileParser():
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()

        ## create a message type map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_type = {name_of:type_of for id_of,name_of,type_of in topics_data}
        self.topic_id = {name_of:id_of for id_of,name_of,type_of in topics_data}
        self.topic_msg_message = {name_of:get_message(type_of) for id_of,name_of,type_of in topics_data}

    def __del__(self):
        self.conn.close()

    # Return [(timestamp0, message0), (timestamp1, message1), ...]
    def get_messages(self, topic_name):

        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        # Deserialise all and timestamp them
        return [ (timestamp,deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp,data in rows]

def read_trajectory(db3_file_path):
    parser = BagFileParser(db3_file_path)
    U = parser.get_messages("/cf231/rpyt")
    X = parser.get_messages("/cf231/pose")
    print(X[0])
    U_new = np.empty((0, 5)) # time, roll, pitch, yaw, thrust
    X_new = np.empty((0, 8)) # time, x, y, z, x_t, y_t, z_t, w
    indices = []
    for i in range(0, len(U)):
        t = U[i][0]
        str_rpyt = U[i][1].data
        l_i = str_rpyt.find('[')
        r_i = str_rpyt.find(']')

        str_rpyt = str_rpyt[l_i+1 : r_i]
        r, p, y, tau = map(float, str_rpyt.split(','))
        U_curr = np.array([[t, r, p, y, tau]])
        U_new = np.append(U_new, U_curr, axis=0)


    for i in range(0, len(X)):
        t = X[i][0]
        x = X[i][1].pose.position.x
        y = X[i][1].pose.position.y
        z = X[i][1].pose.position.z

        x_theta = X[i][1].pose.orientation.x
        y_theta = X[i][1].pose.orientation.y
        z_theta = X[i][1].pose.orientation.z
        w = X[i][1].pose.orientation.w

        X_curr = np.array([[t, x, y, z, x_theta, y_theta, z_theta, w]])
        X_new = np.append(X_new, X_curr, axis=0)
        indices.append(i) 
        # np.save(f"parsed_data/{db3_dirs[c][-18:]}", pose)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_new[:, 1], X_new[:, 2], X_new[:, 3], c=indices, cmap='viridis', label='Trajectory Points')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Plot')

    cbar = plt.colorbar(ax.scatter(X_new[:, 1], X_new[:, 2], X_new[:, 3], c=indices, cmap='viridis'))
    cbar.set_label('Index (i)')

    # plt.savefig(f"{c}.png")
    plt.show()

    return X_new, U_new

def parse_trajectories(db3_file_paths, db3_dirs):
    U_all = []
    X_all = []
    for c, path in enumerate(db3_file_paths):
        X_curr, U_curr = read_trajectory(path)
        U_all.append(U_curr)
        X_all.append(X_curr)
        np.savez(f"UX_{c}.npz", X=X_curr, U=U_curr) # append to a new np array and save that. wq

    U_all = np.array(U_all)
    X_all = np.array(U_all)
    # np.savez(f"UX.npz", U=U_all, X=X_all) # append to a new np array and save that. wq
    # print(U_all.shape, X_all.shape)


if __name__ == "__main__":
        wd = os.getcwd() + '/raw_data/'
        db3_file_paths, db3_dirs = find_db3_files(wd)
        parse_trajectories(db3_file_paths, db3_dirs)
        # read_trajectory(db3_file_paths[0])
        print()