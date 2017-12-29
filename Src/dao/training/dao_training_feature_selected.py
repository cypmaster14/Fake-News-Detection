from Src.db.base import Session, engine, Base
import pandas as pd
import numpy as np
from Src.entities.training_feature_selected import TrainingFeatureSelected


class TrainingFeatureSelectedDAO(object):
    columns_header = ["Date", "Tweet_Text", "Tweet_Id", "User_Id", "User_Name", "User_Screen_Name", "Retweets",
                      "Favorites", "Class", "Tweet_length", "Number_of_URL", "No_of_arond_word", "No_of_hash_word",
                      "Length_of_User_Name", "Number_of_Spam_Word", "Number_of_Swear_Word", "New_Feature"]
    test_feature_selected_file_location = "../Training_feature_selected.csv"

    def __init__(self):
        Base.metadata.create_all(engine)
        pass

    def insert_all(self):
        session = Session()

        test_file = pd.read_csv(self.test_feature_selected_file_location, sep=",", usecols=self.columns_header,
                                index_col=None)
        test_data = np.array(test_file.values[:, :len(self.columns_header)])
        for test in test_data:
            test = test.reshape(1, len(self.columns_header))
            test_entity = TrainingFeatureSelected(
                test[0][0],
                test[0][1],
                test[0][2],
                test[0][3],
                test[0][4],
                test[0][5],
                test[0][6],
                test[0][7],
                test[0][8],
                test[0][9],
                test[0][10],
                test[0][11],
                test[0][12],
                test[0][13],
                test[0][14],
                test[0][15],
                test[0][16],
            )
            session.add(test_entity)

        session.commit()
        session.close()

    def find_all(self):
        session = Session()

        query = [self.get_as_list(entry) for entry in session.query(TrainingFeatureSelected).all()]
        training_feature_selected = pd.DataFrame(query, columns=self.columns_header)
        print(len(training_feature_selected))

        session.close()
        return training_feature_selected

    def get_as_list(self, test_feature_selected: TrainingFeatureSelected) -> list:
        data = test_feature_selected.__dict__
        data = dict(zip(self.columns_header, [data[key] for key in self.columns_header]))

        return list(data.values())


if __name__ == "__main__":
    training_feature_selected_dao = TrainingFeatureSelectedDAO()
    # training_feature_selected_dao.insert_all()
    training_feature_selected_dao.find_all()
