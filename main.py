
# from src.standardize import standardize
# from src.data_model_prep import data_model_prep
# from src.game_grades import game_grades


if __name__ == '__main__':

    ## descriptions in README
    ## import before call so they load in order
    from src.schedules import schedules
    schedules()
    from src.data_inventory import data_inventory
    data_inventory()
    from src.data_download import data_download
    data_download()
    from src.data_process import data_process
    data_process()
    # data_process()
    # standardize()
    # game_grades()
    # data_model_prep()