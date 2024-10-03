import sys
import os

# Add the parent directory (dlatk/) to sys.path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)

from dlatk.featureGetter import FeatureGetter
from dlatk.database.dataEngine import DataEngine
from dlatk.regressionPredictor import RegressionPredictor, ClassifyPredictor
from dlatk.outcomeGetter import OutcomeGetter
import dlatk.dlaConstants as dlac
import DLATKTests
#import ResultHandler as rh
import csv
import subprocess
import multiprocessing
from abc import ABC, abstractmethod



JSON = "DS4UDIsBlackResults2.json"
MODEL_NAME = 'ridgecv'
DATABASE = 'ds4ud_adapt'
TABLES = ['ema_nagg_v9_txt', 'fb_text_v8']
TABLE = 'fb_text_v8'
CORREL_FIELD = 'user_id'
OUTCOME_TABLE = "bl_wave_v9"
OUTCOME_FIELDS = ["phq9_sum"]
OUTCOME_CONTROLS = ["education", "is_male", "individual_income", "age"]
GROUP_FREQ_THRESH = 0
FEATURE_TABLES = [['feat$dr_pca_ema_nagg_v9_txt_reduced100$ema_nagg_v9_txt$user_id'], ['feat$dr_pca_fb_text_v8_reduced100$fb_text_v8$user_id']]#[['feat$cat_LIWC2022$ema_nagg_v9_txt$user_id$1gra'], ['feat$cat_LIWC2022$fb_text_v8$user_id$1gra']]#[['feat$roberta_ba_meL11con$ema_nagg_v9_txt$user_id'], ['feat$roberta_ba_meL11con$fb_text_v8$user_id']]
#FEATURE_TABLES = [['feat$roberta_ba_meL11con$ema_nagg_v9_txt$user_id'], ['feat$roberta_ba_meL11con$fb_text_v8$user_id']]

#python3.5 dlatkInterface.py -d ds4ud_adapt -t ema_nagg_v9_txt -c user_id -f 'feat$roberta_ba_meL11con$ema_nagg_v9_txt$user_id' --fit_reducer --model pca --transform_to_feats ema_nagg_v9_txt_reduced100 --n_components 100
#python3.5 dlatkInterface.py -d ds4ud_adapt -t fb_text_v8 -c user_id -f 'feat$roberta_ba_meL11con$fb_text_v8$user_id' --fit_reducer --model pca --transform_to_feats fb_text_v8_reduced100 --n_components 100



#Grab predictions
#split on types
#get r for each subset



#encode age
def ds4udTests():
    outcome = ["avg_phq9_score"]#["depression_past_any"]
    args = {
        "json" : "DS4UD_ResPatch.json",
        "db" : 'ds4ud_adapt',
        "table" : 'msgs_ema_words_day_v9',
        "correlField" : 'user_id',
        "outcomeTable" : "survey_outcomes_waves_agg_v9_more4_Black_Dep_BinAge",
        "outcomeFields" : outcome,#"avg_phq9_score"],
        "outcomeControls" : [],
        "groupFreqThresh" : 0,
        "featTables" : ['feat$roberta_la_meL23con$msgs_ema_words_day_v9$user_id']
    }

    adaptationFactors = [["age_binarized", "is_female", "is_black"]]#, ["age_binarized"], ["is_female"], ["is_black"]]

    for facs in adaptationFactors:
        
        args["outcomeFields"] = outcome
        args["outcomeControls"] = facs
        DLATKTests.RegressionTest(**args).run()
        #DLATKTests.ClassificationTest(**args).run()
        
        DLATKTests.ResidualControlRegressionTest(**args).run()
        args["outcomeFields"] = outcome + facs
        #args["outcomeControls"] = []
        DLATKTests.FactorAdaptationRegressionTest(**args).run(facs)

        #DLATKTests.FactorAdaptationClassificationTest(**args).run([0])#22])

        #args["outcomeControls"] = facs
        #args["outcomeControls"] = []
        DLATKTests.ResidualFactorAdaptationRegressionTest(**args).run(facs)



def splitOnDemographic():
    outcome = ["depression_past_any"]#["depression_past_any"]
    args = {
        "json" : "DS4UD_Tests8Class.json",
        "db" : 'ds4ud_adapt',
        "table" : 'msgs_ema_words_day_v9',
        "correlField" : 'user_id',
        "outcomeTable" : "survey_outcomes_waves_agg_v9_more4_Black_Dep_BinAge",
        "outcomeFields" : outcome,#"avg_phq9_score"],
        "outcomeControls" : [],
        "groupFreqThresh" : 0,
        "featTables" : ['feat$roberta_la_meL23con$msgs_ema_words_day_v9$user_id']
    }

    adaptationFactors = [["age_binarized", "is_female", "is_black"]]#[["age_binarized", "is_female", "is_black"], ["age_binarized"], ["is_female"], ["is_black"]]

    for facs in adaptationFactors:
        args["outcomeControls"] = facs
        #DLATKTests.ClassificationTest(**args).run()
        DLATKTests.FactorAdaptationClassificationTest(**args).run([0])#22])

def main():

    ds4udTests()

    '''
    adaptationFactors = ["is_black"]

    for index, table in enumerate(TABLES):
        tests.RegressionTest(table = table, featTables=FEATURE_TABLES[index], outcomeFields=["phq9_sum"]).run()
        tests.FactorAdaptationRegressionTest(table = table, featTables=FEATURE_TABLES[index], outcomeControls = OUTCOME_CONTROLS, outcomeFields=["phq9_sum"] + adaptationFactors).run(adaptationFactors=adaptationFactors)
        tests.ResidualFactorAdaptationRegressionTest(table = table, featTables=FEATURE_TABLES[index], outcomeControls = ["is_black"] + OUTCOME_CONTROLS, outcomeFields=["phq9_sum"] + adaptationFactors).run(adaptationFactors=adaptationFactors)

        tests.ClassificationTest(table = table, featTables=FEATURE_TABLES[index], outcomeFields=["depression_past_any"]).run()
        tests.FactorAdaptationClassificationTest(table = table, featTables=FEATURE_TABLES[index], outcomeControls = OUTCOME_CONTROLS, outcomeFields=["depression_past_any"] + adaptationFactors).run(adaptationFactors=adaptationFactors)
    '''


'''
class Test(ABC):
    @abstractmethod
    def __init__(self):
        self.scores = {}
    @abstractmethod
    def run(self):
        pass



class ClassificationTest(Test):

    name = "Classification Test"

    def __init__(self, db=DATABASE, table=TABLE, outcomeTable=OUTCOME_TABLE, correlField = CORREL_FIELD, featTables = FEATURE_TABLES, outcomeFields = OUTCOME_FIELDS, outcomeControls = OUTCOME_CONTROLS, groupFreqThresh = GROUP_FREQ_THRESH):
        og = OutcomeGetter(corpdb = db, corptable = table, correl_field=correlField, outcome_table=outcomeTable, outcome_value_fields=outcomeFields, outcome_controls=outcomeControls, outcome_categories = [], multiclass_outcome = [], featureMappingTable='', featureMappingLex='', wordTable = None, fold_column = None, group_freq_thresh=groupFreqThresh)
        fgs = [FeatureGetter(corpdb = db, corptable = table, correl_field=correlField, featureTable=featTable, featNames="", wordTable = None) for featTable in featTables]

        self.rp = ClassifyPredictor(og, fgs, 'lr', None, None)
        self.result = {
            "name": self.name,
            "tables": {
                "table": table,
                "feat": featTables,
                "outcome": outcomeTable,
                "outcomeFields": outcomeFields,
                "outcomeControls": outcomeControls
            },
            "scores": {}
        }

    def run(self):
        scoresRaw = self.rp.testControlCombos(comboSizes = [], nFolds = 10, allControlsOnly=True)
        self._saveResults(scoresRaw)

    def _saveResults(self, scoresRaw):
        outputStream = open("original_print" + JSON.replace(".json", ".csv"), 'a')
        csv_writer = csv.writer(outputStream)
        csv_writer.writerow([self.name])
        ClassifyPredictor.printComboControlScoresToCSV(scoresRaw, outputStream, delimiter=',')
        outputStream.close()



class FactorAdaptationClassificationTest(ClassificationTest):
    name = "Factor Adaptation Classification Test"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, adaptationFactors = ["logincomeHC01_VC85ACS3yr$10", "hsgradHC03_VC93ACS3yr$10"]):
        
        self.result["adaptationFactors"] = adaptationFactors
        scoresRaw = self.rp.testControlCombos(nFolds = 10, adaptColumns = adaptationFactors, allControlsOnly=True)
        self._saveResults(scoresRaw)



class RegressionTest(Test):

    name = "Regression Test"

    def __init__(self, db=DATABASE, table=TABLE, outcomeTable=OUTCOME_TABLE, correlField = CORREL_FIELD, featTables = FEATURE_TABLES, outcomeFields = OUTCOME_FIELDS, outcomeControls = OUTCOME_CONTROLS, groupFreqThresh = GROUP_FREQ_THRESH):
        og = OutcomeGetter(corpdb = db, corptable = table, correl_field=correlField, outcome_table=outcomeTable, outcome_value_fields=outcomeFields, outcome_controls=outcomeControls, outcome_categories = [], multiclass_outcome = [], featureMappingTable='', featureMappingLex='', wordTable = None, fold_column = None, group_freq_thresh=groupFreqThresh)
        fgs = [FeatureGetter(corpdb = db, corptable = table, correl_field=correlField, featureTable=featTable, featNames="", wordTable = None) for featTable in featTables]
        RegressionPredictor.featureSelectionString = dlac.DEF_RP_FEATURE_SELECTION_MAPPING['magic_sauce']

        self.rp = RegressionPredictor(og, fgs, MODEL_NAME, None, None)
        self.result = {
            "name": self.name,
            "tables": {
                "table": table,
                "feat": featTables,
                "outcome": outcomeTable,
                "outcomeFields": outcomeFields,
                "outcomeControls": outcomeControls
            },
            "scores": {}
        }

    def run(self):
        scoresRaw = self.rp.testControlCombos(comboSizes = [], nFolds = 10, allControlsOnly=True, report = False)
        self._saveResults(scoresRaw)

    def _saveResults(self, scoresRaw):
        outputStream = open("original_print" + JSON.replace(".json", ".csv"), 'a')
        csv_writer = csv.writer(outputStream)
        csv_writer.writerow([self.name])
        RegressionPredictor.printComboControlScoresToCSV(scoresRaw, outputStream)
        outputStream.close()



class ResidualControlRegressionTest(RegressionTest):

    name = "Residualized Controls Regression Test"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        scoresRaw = self.rp.testControlCombos(nFolds = 10, residualizedControls =True, allControlsOnly=True, report = False)
        self._saveResults(scoresRaw)



class FactorAdaptationRegressionTest(RegressionTest):
    name = "Factor Adaptation Regression Test"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, adaptationFactors = ["logincomeHC01_VC85ACS3yr$10", "hsgradHC03_VC93ACS3yr$10"]):
        self.result["adaptationFactors"] = adaptationFactors
        scoresRaw = self.rp.testControlCombos(nFolds = 10, adaptationFactorsName = adaptationFactors, integrationMethod="fa", allControlsOnly=True, report = False)
        self._saveResults(scoresRaw)
        


class ResidualFactorAdaptationRegressionTest(RegressionTest):
    name = "Residualized Factor Adaptation Regression Test"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, adaptationFactors = ["logincomeHC01_VC85ACS3yr$10", "hsgradHC03_VC93ACS3yr$10"]):
        self.result["adaptationFactors"] = adaptationFactors
        scoresRaw = self.rp.testControlCombos(nFolds = 10, adaptationFactorsName = adaptationFactors, residualizedControls =True, integrationMethod="rfa", allControlsOnly=True, report = False)
        self._saveResults(scoresRaw)

'''

if __name__ == "__main__":
    main()
