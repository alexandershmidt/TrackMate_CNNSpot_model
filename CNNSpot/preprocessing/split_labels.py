from sklearn.model_selection import train_test_split
def split_labels(train_x, train_y, test_size):
    train_x_split, test_x_split, train_y_split, test_y_split = train_test_split(train_x, train_y, test_size=test_size)
    return train_x_split.transpose(), train_y_split.transpose(), test_x_split.transpose(), test_y_split.transpose()