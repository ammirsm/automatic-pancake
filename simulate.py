import os
import uuid
import warnings
from datetime import datetime

from app.agent import ActiveLearningAgent
from app.configs import Config, load_from_json
from app.data import Data
from app.import_export import export_json
from app.model import LearningModel

warnings.filterwarnings("ignore")


def export_json_file(dirs, file_name, the_data):
    path = Config.result + "/".join(dirs)
    os.makedirs(path, exist_ok=True)
    export_json(the_data, f"{path}/{file_name}.json")


if __name__ == "__main__":
    configs = load_from_json("./app/configs.json")
    first_time = datetime.now()
    print("START", flush=True)
    for the_config in configs:
        for iteration in range(the_config.number_of_iterations):
            tic = datetime.now()
            # create and initial Data object
            data_obj = Data(**the_config.data)
            data_obj.init_data()
            if len(data_obj.label_unique_values()) != 2:
                del data_obj
                print("Not binary classification")
                continue
            print(
                "-------\n",
                the_config.key,
                flush=True,
            )
            feature_config = the_config.feature_config

            # Create FeatureExtraction object
            FeatureExtractorClass = feature_config.get("feature_extractor_class")
            feature_extractor = FeatureExtractorClass(
                **the_config.feature_extractor_data
            )

            # Create Learning model object
            learning_model = LearningModel(
                data=data_obj,
                feature_extractor=feature_extractor,
                **the_config.learning_model_data,
            )

            # Create QueryStrategy object
            QueryStrategyClass = the_config.strategy.get("strategy_class")
            query_strategy = QueryStrategyClass(
                learning_model=learning_model, **the_config.query_strategy_data
            )

            # Create agent object and run the active learning function
            agent = ActiveLearningAgent(
                query_strategy=query_strategy,
                learning_model=learning_model,
                **the_config.agent_data,
            )

            agent.start_active_learning()

            # get export data and making ready for export
            export_data = the_config.export_data
            export_data["plot_data"] = agent.plot_data()
            export_data["times"] = agent.times_spent
            export_data["number_of_papers"] = data_obj.number_of_papers
            export_data["number_of_relavant"] = data_obj.number_of_relavant
            output_dir = the_config.output_dir
            output_dir.insert(0, Config.main_directory_name)
            export_json_file(
                output_dir,
                str(uuid.uuid4().hex),
                export_data,
            )

            print("time: ", datetime.now() - tic, flush=True)

    print("-------\ntotal time spend: ", datetime.now() - first_time)
    print(
        "-----------------------------------------END----------------------------------------",
        flush=True,
    )
