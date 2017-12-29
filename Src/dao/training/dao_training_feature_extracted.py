from Src.db.base import Session, engine, Base
import pandas as pd
import numpy as np
from Src.entities.training_feature_extracted import TrainingFeatureExtracted


class TrainingFeatureExtractedDAO(object):
    test_feature_extracted_file_location = "../../Training_feature_extracted.csv"
    columns_header = ['Retweets', 'Favorites', 'New_Feature', 'Class']

    def __init__(self):
        Base.metadata.create_all(engine)
        pass

    def insert_all(self):
        session = Session()

        test_file = pd.read_csv(self.test_feature_extracted_file_location, sep=",", usecols=self.columns_header,
                                index_col=None)
        test_data = np.array(test_file.values[:, :4])
        for test in test_data:
            test = test.reshape(1, 4)
            test_entity = TrainingFeatureExtracted(test[0][0], test[0][1], test[0][2], test[0][3])
            session.add(test_entity)

        session.commit()
        session.close()

    def findAll(self):
        session = Session()

        query = [self.get_as_list(entry) for entry in session.query(TrainingFeatureExtracted).all()]
        training_feature_extracted = pd.DataFrame(query, columns=self.columns_header)
        print(len(training_feature_extracted))

        session.close()
        return training_feature_extracted

    def get_as_list(self, test_feature_extracted: TrainingFeatureExtracted) -> list:
        data = test_feature_extracted.__dict__
        data = dict(zip(self.columns_header, [data[key] for key in self.columns_header]))
        return list(data.values())


if __name__ == "__main__":
    training_feature_extracted_dao = TrainingFeatureExtractedDAO()
    # training_feature_extracted_dao.insert_all()
    training_feature_extracted_dao.findAll()
