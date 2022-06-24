from Database.database_interface import MySqlInterface
from data_pipeline import pipeline
host="127.0.0.1"
user="root"
password="Ol1v3rGra1ng3"

sql = MySqlInterface(host, user, password)
pipeline = pipeline()
pipeline.fit_transform()
print(pipeline.raw_input.shape, pipeline.raw_output.shape)

sample_dict_name = "samples"
samples_dict = {"sampleID":"int PRIMIARY KEY",
                 "inputID":"int UNIQUE",
                 "outputID":"int UNIQUE"}

input_dict_name = "inputs"
input_dict = {"ID":"int NOT NULL AUTOINCREMENT PRIMARY KEY",
              "inputID":"int FOREIGN KEY REFERENCES samples(inputID)",
              "feature1ID":"int UNIQUE",
              "feature2ID":"int UNIQUE",
              "feature3ID":"int UNIQUE",
              "feature4ID":"int UNIQUE",
              "feature5ID":"int UNIQUE",
              "feature6ID":"int UNIQUE",
              "feature7ID":"int UNIQUE"}

output_dict_name = "outputs"
output_dict = {"ID":"int NOT NULL AUTOINCREMENT PRIMARY KEY",
              "outputID":"int FOREIGN KEY REFERENCES samples(outputID)",
              "feature1":"int UNIQUE",
              "feature2":"int UNIQUE",
              "feature3":"int UNIQUE"}

input_f1_name = "input_f1"
input_f1_name = {"ID":"int NOT NULL AUTOINCREMENT PRIMARY KEY",
                 "feature1ID":"int FOREIGN KEY REFERENCES inputs(feature1ID)"}
for time in np.arange(241):
    t_string = "T" + str(int(time))
    input_f1_name[t_string] = "FLOAT NOT NULL"

input_f2_name = "input_f2"
input_f2_name = {"ID": "int NOT NULL AUTOINCREMENT PRIMARY KEY",
                 "feature2ID": "int FOREIGN KEY REFERENCES inputs(feature2ID)"}
for time in np.arange(241):
    t_string = "T" + str(int(time))
    input_f1_name[t_string] = "FLOAT NOT NULL"

input_f1_name = "input_f1"
input_f1_name = {"ID":"int NOT NULL AUTOINCREMENT PRIMARY KEY",
                 "feature1ID":"int FOREIGN KEY REFERENCES inputs(feature1ID)"}
for time in np.arange(241):
    t_string = "T" + str(int(time))
    input_f1_name[t_string] = "FLOAT NOT NULL"

input_f1_name = "input_f1"
input_f1_name = {"ID":"int NOT NULL AUTOINCREMENT PRIMARY KEY",
                 "feature1ID":"int FOREIGN KEY REFERENCES inputs(feature1ID)"}
for time in np.arange(241):
    t_string = "T" + str(int(time))
    input_f1_name[t_string] = "FLOAT NOT NULL"




