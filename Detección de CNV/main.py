from gui import Aplication
from controller import Controller
from utils import load_model


class AppDet():

    def __init__(self):
        print('Loading model ...')
        model = load_model()
        print('Model loaded.')
        app = Aplication()
        con = Controller(app, model)


def main():
    appDet = AppDet()


if __name__ == '__main__':
    main()
