
from malware_detector import Malware_Detector
import os

DIRECTORY_PATH = '../'
data_path = 'data.csv'
target_classes = {
    'Benign': 'Benign',
    'Malware': {
        'Trojan': {
            'Emotet': 'Emotet',
            'Reconyc': 'Reconyc',
            'Refroso': 'Refroso',
            'Scar': 'Scar',
            'Zeus': 'Zeus',
            'model': ['xgb','random_forest']
        },
        'Spyware': {
            '180solutions': '180solutions',
            'CWS': 'CWS',
            'Gator': 'Gator',
            'TIBS': 'TIBS',
            'Transponder': 'Transponder',
            'model': ['xgb','random_forest']
        },
        'Ransomware': {
            'Ako': 'Ako',
            'Conti': 'Conti',
            'Maze': 'Maze',
            'Pysa': 'Pysa',
            'Shade': 'Shade',
            'model': ['xgb','random_forest']
        },
        'model': ['cart', 'random_forest']
    },
    'model': ['svm']
}

malware_detector = Malware_Detector(path=data_path, target_classes=target_classes, malware_features=['Class', 'malware_cat', 'malware_fam'], classification_level=1, prediction='max')

malware_detector.fit()

malware_detector.report_()