from sqlalchemy import Column, String, Integer, Float
from Src.db.base import Base


class TrainingFeatureSelected(Base):
    __tablename__ = 'training_feature_selected'

    id = Column(Integer, primary_key=True)
    Date = Column(String)
    Tweet_Text = Column(String)
    Tweet_Id = Column(String)
    User_Id = Column(String)
    User_Name = Column(String)
    User_Screen_Name = Column(String)
    Retweets = Column(Float)
    Favorites = Column(Float)
    Class = Column(Float)
    Tweet_length = Column(Integer)
    Number_of_URL = Column(Integer)
    No_of_arond_word = Column(Integer)
    No_of_hash_word = Column(Integer)
    Length_of_User_Name = Column(Integer)
    Number_of_Spam_Word = Column(Integer)
    Number_of_Swear_Word = Column(Integer)
    New_Feature = Column(Integer)

    def __init__(self,
                 Date,
                 Tweet_Text,
                 Tweet_Id,
                 User_Id,
                 User_Name,
                 User_Screen_Name,
                 Retweets,
                 Favorites,
                 Class,
                 Tweet_length,
                 Number_of_URL,
                 No_of_arond_word,
                 No_of_hash_word,
                 Length_of_User_Name,
                 Number_of_Spam_Word,
                 Number_of_Swear_Word,
                 New_Feature):
        self.Date = Date
        self.Tweet_Text = Tweet_Text
        self.Tweet_Id = Tweet_Id
        self.User_Id = User_Id
        self.User_Name = User_Name
        self.User_Screen_Name = User_Screen_Name
        self.Retweets = Retweets
        self.Favorites = Favorites
        self.Class = Class
        self.Tweet_length = Tweet_length
        self.Number_of_URL = Number_of_URL
        self.No_of_arond_word = No_of_arond_word
        self.No_of_hash_word = No_of_hash_word
        self.Length_of_User_Name = Length_of_User_Name
        self.Number_of_Spam_Word = Number_of_Spam_Word
        self.Number_of_Swear_Word = Number_of_Swear_Word
        self.New_Feature = New_Feature
