from sqlalchemy import Column, Integer, Float
from Src.db.base import Base


class TrainingFeatureExtracted(Base):
    __tablename__ = 'train_feature_extracted'

    id = Column(Integer, primary_key=True)
    Retweets = Column(Integer)
    Favorites = Column(Float)
    New_Feature = Column(Integer)
    Class = Column(Integer)

    def __init__(self, ReTweets, Favorites, New_Feature, Class):
        self.Retweets = ReTweets
        self.Favorites = Favorites
        self.New_Feature = New_Feature
        self.Class = Class
