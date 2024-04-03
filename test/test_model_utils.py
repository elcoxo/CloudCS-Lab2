# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from model_utils import make_inference, load_model
from sklearn.pipeline import Pipeline
from pickle import dumps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler


@pytest.fixture
def create_data() -> dict[str, int | float]:
    return {"test": 'hi my name is slim shady'}


@pytest.fixture()
def filepath_and_data(tmpdir):
    p = tmpdir.mkdir("datadir").join("fakedmodel.pkl")
    example: str = "Test message!"
    p.write_binary(dumps(example))
    return str(p), example


def test_load_model(filepath_and_data):
    assert filepath_and_data[1] == load_model(filepath_and_data[0])


def test_make_inference(monkeypatch, create_data):
    def mock_get_predictions(_):
        # assert create_data == {
        #     key: value[0] for key, value in data.to_dict("list").items()
        # }
        return {"IE": 1.0,
                "NS": 1.0,
                "TF": 0.0,
                "JP": 0.0
                }

    def mock_get_transform(_):
        corpus = ['This is the first document.',
                  'This document is the second document.']
        vectorizer = TfidfVectorizer()
        create_tranform = vectorizer.fit_transform(corpus)

        return create_tranform

    in_model: Pipeline = [{'Personality': Pipeline}, TfidfVectorizer()]
    monkeypatch.setattr(in_model[1], "transform", mock_get_transform)
    monkeypatch.setattr(Pipeline, "predict", mock_get_predictions)

    print(mock_get_predictions)
    print(in_model)
    print(create_data)

    result = make_inference(in_model, create_data)
    assert result == {'Personality': {"IE": 1.0,
                      "NS": 1.0,
                                      "TF": 0.0,
                                      "JP": 0.0
                                      }
                      }
