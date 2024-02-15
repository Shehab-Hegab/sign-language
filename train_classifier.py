import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data_two.pickle', 'rb'))

# Filter data and labels with shape 84
filtered_indices = [idx for idx, item in enumerate(data_dict['data']) if np.asarray(item).shape == (84,)]

data = np.asarray([data_dict['data'][idx] for idx in filtered_indices])
labels = np.asarray([data_dict['labels'][idx] for idx in filtered_indices])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples with shape (84,) were classified correctly!'.format(score * 100))

f = open('model_two.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
