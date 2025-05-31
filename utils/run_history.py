def _get_run_history(automl_model):
    """Get the run history from the database."""
    # Access the RunHistory
    data_for_json = []
    run_history = automl_model.automl_.runhistory_

    for run_key, run_value in run_history.data.items():
        config_id = run_key.config_id
        config = run_history.ids_config[config_id]

        config_dict = config.get_dictionary()

        data_for_json.append(
            {
                "config_id": config_id,
                "configuration": config_dict,
                "cost": run_value.cost,
                "time": run_value.time,
                "additional_info": run_value.additional_info,
            }
        )

    return data_for_json
