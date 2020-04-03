import sys

from LogicComponents import *


class VisualizationApp(QMainWindow):
    def __init__(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Spúšťacia trieda. Je obalom celeje aplikácie. Vytvorí programové okno, na celú obrazovku. Táto trieda dedí od
        objektu Tk z build in knižnice tkinter.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.frames: je to dictionary, ktorý obsahuje jednotlivé hlavné stránky aplikácie. Kľúčom k týmto stránkam
                          sú prototypy daných stránok
        """
        # Konštruktor základnej triedy.
        super(QMainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle('Bakalarka')
        self.setContentsMargins(0, 0, 0, 0)
        graphPage = GraphPage()
        self.setCentralWidget(graphPage)


class GraphPage(QWidget):
    def __init__(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Podstránka, na ktorej je zobrazovaný rámec s grafmi, váhami a okno s detailmi vrstvy.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__logic_layer: Referencia na logiku aplikácie.
        :var self.__keras_model: Načítaný model, jeho referencia je zaslaná do logic layer, kde sa menia váhy.
                                 Používa sa aj pri ukladaní.
        :var self.__file_path:   Cesta k súboru pre lepší konfort pri načítaní a ukladaní.
        :var self.__file_name:   Meno súboru pre lepší konfort pri ukladaní.
        :var self.__info_label:
        :var self.__mg_frame:
        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param parent: nadradený tkinter Widget
        """
        super().__init__(*args, **kwargs)
        self.__file_path = '.'
        self.__file_name = ''
        self.__keras_model = None
        self.__logic_layer = None
        self.__mg_frame = MainGraphFrame()
        self.__open_load_info_label = QLabel()
        self.__save_model_info_label = QLabel()

        self.initialize_ui()

        self.__logic_layer = GraphLogicLayer(self.__mg_frame)

        # self.open_model('./modelik.h5')
        #self.open_model('./cnn_mnist.h5')
        self.open_model('./small_1channel.h5')
        # self.open_model('./small_cnn.h5')
        # self.load_points('./2d_input.txt')

    def initialize_ui(self):
        vertical_layout = QVBoxLayout()
        vertical_layout.setContentsMargins(0, 0, 0, 0)
        vertical_layout.setSpacing(0)
        button_wrapper = QWidget()
        button_wrapper.setFixedHeight(50)
        vertical_layout.addWidget(button_wrapper)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(0)

        open_load_layout = QHBoxLayout()
        open_load_layout.setSpacing(5)
        open_load_layout.setContentsMargins(0, 0, 0, 0)

        group_wrapper_open_load = QWidget()

        group_wrapper_save = QWidget()

        self.__open_load_info_label.setText("<font color='orange'>Load keras model.</font>")

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.__open_load_info_label.setFont(font)
        self.__save_model_info_label.setFont(font)

        # Button layout
        button_layout.setContentsMargins(0, 0, 0, 0)

        # Open model button
        open_btn = QPushButton('Open model')
        open_btn.clicked.connect(self.try_open_model)
        open_load_layout.addWidget(open_btn)

        # Load points button
        load_btn = QPushButton('Load points')
        load_btn.clicked.connect(self.try_load_points)
        open_load_layout.addWidget(load_btn)

        # Load labels button
        load_labels_btn = QPushButton('Load labels')
        load_labels_btn.clicked.connect(self.try_load_labels)
        open_load_layout.addWidget(load_labels_btn)
        open_load_layout.addWidget(self.__open_load_info_label, alignment=QtCore.Qt.AlignRight)

        # Save model button
        save_btn = QPushButton('Save model')
        save_btn.clicked.connect(self.save_model)

        # Save model layout
        save_layout = QHBoxLayout()
        save_layout.setSpacing(5)
        save_layout.setContentsMargins(0, 0, 0, 0)
        save_layout.addWidget(self.__save_model_info_label)
        save_layout.addWidget(save_btn)

        group_wrapper_save.setLayout(save_layout)

        group_wrapper_open_load.setLayout(open_load_layout)

        button_layout.addWidget(group_wrapper_open_load, alignment=QtCore.Qt.AlignLeft)

        button_layout.addWidget(group_wrapper_save, alignment=QtCore.Qt.AlignRight)

        button_wrapper.setLayout(button_layout)

        vertical_layout.addWidget(self.__mg_frame)

        self.setLayout(vertical_layout)

    def try_open_model(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Načítanie vybraného modelu, pokiaľ bola zvolená nejaká cesta k súboru.
        """
        file_path = QFileDialog.getOpenFileName(self, 'Load keras model', self.__file_path, 'Keras model (*.h5)')[0]
        if file_path != '':
            self.open_model(file_path)

    def try_load_labels(self):
        if self.__keras_model is not None:
            file_path = QFileDialog.getOpenFileName(self, 'Load labels', self.__file_path, 'Text or CSV files (*.txt *.csv)')[0]
            if file_path != '':
                error_message = self.__logic_layer.load_labels(file_path)
                if error_message is not None:
                    self.__open_load_info_label.setText("<font color='red'>{}</font>".format(error_message))
                    self.__open_load_info_label.show()
                else:
                    self.__open_load_info_label.hide()
        else:
            self.__open_load_info_label.setText("<font color='red'>You have to load model first!</font>")

    def open_model(self, filepath):
        """
        Popis:
        ----------------------------------------------------------------------------------------------------------------
        Rozdelenie zadanej cesty k súboru na absolútnu cestu a názov súboru.
        Načítanie súboru na základe cesty.
        Inicializácia logickej vrstvy, načítaným modelom.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param filepath: aboslutná cesta k súboru.
        """
        self.__file_path, self.__file_name = ntpath.split(filepath)
        self.__keras_model = keras.models.load_model(filepath)
        print(self.__keras_model.input_shape)
        self.__logic_layer.initialize(self.__keras_model)
        self.__open_load_info_label.hide()
        self.__save_model_info_label.hide()

    def try_load_points(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Získanie adresy k súboru s hodnotami bodov. Ak nie je načítaný model, zobrazí sa informácia o chybe.
        """
        if self.__keras_model is not None:
            input_shape = self.__keras_model.input_shape
            file_path = ''
            if len(input_shape) > 2:
                msgBox = CustomMessageBox('Výber vstupu', 'Ako si prajete zvoliť vstup?', 'Jednotlivo', 'Priečinok',
                                          'Zrušiť', 'Jednotlivo: manuálny výber viacerých obrázkov ' +
                                          '\nPriečinok: priečinok obshujúci priečinky so vstupmi (priradí label)')

                res = msgBox.exec_()

                if msgBox.result() == 0:
                    file_path = QFileDialog.getOpenFileNames(self, 'Load model input', self.__file_path,
                                                             'Image files (*.jpg *.png )')
                    if len(file_path[0]) != 0:
                        error_message = self.__logic_layer.handle_images_input(file_path[0])
                elif msgBox.result() == 1:
                    file_path = QFileDialog.getExistingDirectory(self, 'Zvoľte priečinok', self.__file_path)
                    error_message = self.__logic_layer.handle_images_input(file_path)
                else:
                    return
                print(file_path)
                return
            else:
                file_path = QFileDialog.getOpenFileName(self, 'Load model input', self.__file_path,
                                                        'Text files (*.txt);;CSV files (*.csv)')[0]
            if file_path != '':
                self.load_points(file_path)
        else:
            self.__open_load_info_label.setText("<font color='red'>You have to load model first!</font>")

    def load_points(self, filepath: str):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Načítanie bodov do modelu na základe cesty k súboru so vstupnými dátami. Ak pri načítaní bodov dôjde ku chybe,
        je táto chyba zobrazená.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param filepath: cesta k súboru
        :type filepath:  str
        """
        error_message = self.__logic_layer.load_points(filepath)
        if error_message is not None:
            self.__open_load_info_label.setText("<font color='red'>{}</font>".format(error_message))
            self.__open_load_info_label.show()
        else:
            self.__open_load_info_label.hide()

    def save_model(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Uloženie modelu.
        """
        if self.__keras_model is not None:
            file_path = ''
            QFileDialog.getSaveFileName(self, 'Save keras model', ntpath.join(self.__file_path, self.__file_name),
                                        'Keras model (*.h5)')[0]
            if file_path != '':
                self.__file_path, self.__file_name = ntpath.split(file_path)
                self.__keras_model.save(file_path)
            self.__save_model_info_label.hide()
        else:
            self.__save_model_info_label.show()
            self.__save_model_info_label.setText('You have to load model first!')
            self.__save_model_info_label.setStyleSheet("QLabel { color : red; }")


class MainGraphFrame(QWidget):
    """
    Popis
    --------------------------------------------------------------------------------------------------------------------
    Hlavné okno podstránky s grafmi. Okno vytvorí scrollovacie okno, do ktorého budú následne vkladané jednotlivé vrstvy
    s grafmi a ich ovládačmi. Vytvorí taktiež options okno, ktoré slúži na zobrazovanie informácii o zvolenem bode a
    možnosti nastavenia zobrazenia grafov v jednotlivých vrstvách.

     Atribúty
     -------------------------------------------------------------------------------------------------------------------
     :var self.__logic_layer:        obsahuje odkaz na vrstvu, ktorá sa zaoberá výpočtami a logickou reprezentáciou
                                     jednotlivých vrstiev a váh.

     :var self.__number_of_layers:   počet vrstiev načítanej neurónovej siete

     :var self.__name_to_order_dict: každej vrstve je postupne prideľované poradové číslo vrstvy, tak ako ide od
                                     začiatku neurónovej siete. Ako kľúč je použitý názov vrstvy.

     :var self.__order_to_name_dict: podľa poradia je každej vrstve pridelené poradové číslo. Poradové číslo je kľúčom
                                     do dict. Jeho hodnotami sú názvy jedntlivých vrstiev.
                                     Ide o spätný dict k __name_to_order_dict.

     :var self.__active_layers:      obsahuje poradové čísla aktívnych vrstiev, pre prídavanie a odoberanie vrstiev z
                                     add_graph_fame

     :var self.__input_panned:       obal pre komponenty InputDataFrame a PannedWindow. Umožňuje podľa potreby
                                     roztiahnuť alebo zúžiť veľkosť input panelu na úkor PannedWindow, ktoré obsahuje
                                     rámce s reprezentáciou jednotlivých vrstiev.

     :var self.__graph_panned:       obal pre ScrollableWindow, ktoré obsahuje rámce v ktorých sú zobrazené jednotlivé
                                     vrstvy.

     :var self.__options_frame:      okno, v ktorom sa zobrazujú možnosti pre zvolenú vrstvu.

     :var self.__scroll_frame:       obsahuje grafické zobrazenie zvolených vrstiev neurónovej siete a
                                     ComboboxAddRemoveFrame, v ktorom sú volené vrstvy, ktoré chceme zobraziť

     :var self.__add_graph_frame:    combobox, z ktorého si vyberáme jednotlivé, ešte neaktívne vrstvy, ktoré tlačidlom
                                     následne zobrazíme. Zobrazuje sa, len ak nie sú zobrazené ešte všetky vrstvy.
     """

    def __init__(self, *args, **kwargs):
        """
        :param parent: nadradený tkinter widget
        :type parent: tk.Widget
        :param logic_layer: odkaz na logickú vrstvu
        :type logic_layer: GraphLogicLayer
        """
        super().__init__(*args, **kwargs)

        # self.__logic_layer = logic_layer
        self.__logic_layer = None
        self.__number_of_layers = 0

        self.__neural_layers = None

        self.__name_to_order_dict = {}

        self.__order_to_name_dict = {}

        self.__active_layers = []

        # Grafické komponenty.
        self.scrollable_frame = QScrollArea()

        self.__scrollbar_content = QWidget()

        self.__graph_area_layout = QHBoxLayout()

        self.__options_frame = OptionsFrame()

        self.__add_remove_layer_rc = RemovingCombobox()

        self.initialize_ui()

        self.__options_frame.hide_option_bar()

        # layer_graph.clear()
        # layer_graph.hide()
        # self.__graph_area_layout.removeWidget(layer_graph)

    def initialize_ui(self):
        horizontal_layout = QHBoxLayout()
        horizontal_layout.setContentsMargins(0, 0, 5, 0)
        horizontal_layout.setSpacing(0)

        self.__options_frame.setFixedWidth(320)
        self.__options_frame.setSizePolicy(Qt.QSizePolicy.Minimum, Qt.QSizePolicy.Minimum)

        self.scrollable_frame.setWidgetResizable(True)
        horizontal_layout.addWidget(self.__options_frame, alignment=QtCore.Qt.AlignLeft)
        horizontal_layout.addWidget(self.scrollable_frame)
        self.__options_frame.adjustSize()

        self.__graph_area_layout.setContentsMargins(0, 0, 0, 0)
        self.__graph_area_layout.setSpacing(0)
        self.__graph_area_layout.setAlignment(QtCore.Qt.AlignLeft)
        self.__scrollbar_content.setLayout(self.__graph_area_layout)

        self.scrollable_frame.setWidget(self.__scrollbar_content)

        self.__add_remove_layer_rc.setFixedWidth(412)

        self.__graph_area_layout.addWidget(self.__add_remove_layer_rc)

        self.setLayout(horizontal_layout)

    def initialize(self, logic_layer: GraphLogicLayer):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializačná funkcia. Inicializuje všetky potrbné komponenty tejto vrstvy.
        Vytvorí predbežný zoznam názvov jednotlivých vrstiev. Po inicializácií AddRemoveComboboxFrame budú získane
        unikátne mená, ktoré sa použijú ako kľúče v slovníku.

        Na základe unikátnych mien je vytvorený slovník, ktorý prevádza meno vrstvy na jej poradové číslo a spätný
        slovník, ktorý prevádza poradové číslo na názov vrstvy.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param neural_layers: zoznam NeuralLayer, ktoré bude možné zobraziť.
        :param active_layer:  refenrencia na zoznam aktívnych vrstiev.
        """
        self.__options_frame.initialize()

        # Vyčistenie slovníkov.
        self.__logic_layer = logic_layer
        self.__order_to_name_dict = {}
        self.__name_to_order_dict = {}

        self.__active_layers = logic_layer.active_layers

        self.__neural_layers = logic_layer.neural_layers
        self.__number_of_layers = len(self.__neural_layers)

        # Vytvorenie predbežného zoznamu názvov vrstiev.
        layer_name_list = []
        for i in range(self.__number_of_layers):
            layer_name_list.append(self.__neural_layers[i].layer_name)

        # Inicializácia AddRemoveComboboxFrame. Funkcia navracia list unikátnych názvov vrstiev.
        # Unikátne názvy su použité ako kľúče v slovníku.
        unique_name_list = self.__add_remove_layer_rc.initialize(layer_name_list, self.show_layer, 'Add layer', True,
                                                                 'Select layer')
        for i, layerName in enumerate(unique_name_list):
            self.__neural_layers[i].layer_name = layerName
            self.__order_to_name_dict[i] = layerName
            self.__name_to_order_dict[layerName] = i

        if len(self.__active_layers) < self.__number_of_layers:
            self.__add_remove_layer_rc.show()
        else:
            self.__add_remove_layer_rc.hide()

        first_layer_tuple = (self.__neural_layers[0].layer_number, self.__neural_layers[0].layer_name)
        self.show_layer(first_layer_tuple)

    def show_layer(self, layer_tuple: tuple):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda na základe parametra obdržaného z triedy AddRemoveCombobox vytvorí a následne zobrazí zvolenú vrstvu.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layer_tuple:
        (poradové číslo vrstvy ,názov vrstvy) - obashuje hodnotu z triedy AddRemoveCombobox
        """
        layer_name = layer_tuple[1]
        layer_number = self.__name_to_order_dict[layer_name]

        # Otestuje, či je číslo vrstvy valídne.
        if 0 <= layer_number < self.__number_of_layers:
            layer_to_show = None

            # Inicializácia vrstvy na zobrazenie.
            if layer_number < self.__number_of_layers:
                layer_to_show = self.__neural_layers[layer_number]
                graph_frame_widget = layer_to_show.create_graph_frame(self.show_layer_options_frame, self.hide_layer)
                graph_frame_widget.setMinimumWidth(412)
                self.__graph_area_layout.insertWidget(self.__graph_area_layout.count() - 1, graph_frame_widget)

            # Poradové číslo vrstvy je vložené do listu aktívnych vrstiev, ktorý sa využíva pri efektívnejšom updatovaní
            # vykresľovaných grafov.
            self.__active_layers.append(layer_number)
            self.__add_remove_layer_rc.hide_item(layer_name)

            self.__logic_layer.set_points_for_layer(layer_number)
            layer_to_show.apply_changes()
            layer_to_show.use_config()
            layer_to_show.redraw_graph_if_active()
            # Ak je počet aktívnych vrstiev rovný celkovému počtu vrstiev je skrytý panel pre pridávanie nových vrstiev.
            if len(self.__active_layers) == self.__number_of_layers:
                self.__add_remove_layer_rc.hide()

    def hide_layer(self, layer_number: int):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Skryje vrstvu, podľa jej poradového čísla.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layer_number: číslo vrstvy, ktorá má byť skrytá
        :type layer_number: int
        """
        if layer_number in self.__active_layers:
            layer_name = self.__order_to_name_dict[layer_number]
            layer = self.__neural_layers[layer_number]
            layer.graph_frame.clear()
            layer.graph_frame = None

            self.__active_layers.remove(layer_number)
            self.__add_remove_layer_rc.show_item(layer_name)
            if len(self.__active_layers) < self.__number_of_layers:
                self.__add_remove_layer_rc.show()
            if self.__options_frame.active_layer == layer:
                self.__options_frame.hide_option_bar()

    def show_layer_options_frame(self, layer_number):
        if 0 <= layer_number < self.__number_of_layers:
            layer = self.__neural_layers[layer_number]
            self.__options_frame.initialize_with_layer_config(layer, layer.config)

    def apply_changes_on_options_frame(self):
        self.__options_frame.update_selected_config()

    def update_active_options_layer(self, start_layer):
        self.__options_frame.update_active_options_layer(start_layer)

    def update_layer_if_active(self, neural_layer_ref):
        if neural_layer_ref == self.__options_frame.active_layer:
            self.apply_changes_on_options_frame()

class OptionsFrame(QWidget):
    def __init__(self, *args, **kwargs):
        """"
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Obsahuje ovladacie prvky pre jednotlivé vrstvy. V rámci nej je možne navoliť zobrazované súradnice pre
        jednotlivé metódy, ofarbiť body na základe ich label, povoliť zobrazenie mriežky, ak je to možné, uzmaknutie
        pohľadu v grafe.
        Je možné aj zvoliť redukciu priestoru a zobraziť požadovaný PCA komponent alebo použiť metódu t-SNE.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__graph_logic:          odkaz na triedu, zaoberajúcu sa logikou aplikácie.
        :var self.__labels_entries_list:  list vstupnov pre zadávanie názvov osi v grafe aktívnej vrstvy.
        :var self.__cords_entries_list:   list vstupov na zadávanie zobrazovaných súradníc
        :var self.__bar_wrapper:          obaľovací widget, stále zobrazený
        :var self.__layer_options_frame:  obaľovací widget, ktorý obaľuje jedntolivé ucelené rámce s možnosťami
        :var self.__layer_name_label:     zobrazuje meno aktívnej vrstvy, pre ktorú sú zobrazované možnosti
        :var self.__cords_choose_frame:   rámec obsahujúci možnosti zobrazovaných súradníc
        :var self.__possible_cords_label: zobrazuje informáciu o rozsahu súradníc, ktoré môžu byť zobrazené.
        :var self.__label_choose_frame:   obaľuje widgety pre vsutpy, ktoré sú použité na načítanie názvu osí grafu
        :var self.__appearance_frame:     obľuje checkboxy, ktoré slúžia na úpravu vzhľadu grafu a jeho správania
                                          pri prekreslení
        :var self.__color_labels:         boolean premenná checboxu, ktorá označuje, či majú byť vstupy s
                                          rovnakým labelom, ofarbené rovnakou farbou
        :var self.__color_labels_check:   checkbox, ktorý zachytáva, či je táto možnosť zvolená alebo nie
        :var self.__lock_view:            boolean premenná checkboxu, ktorá určuje či má graf pri prekresľovaní zmeniť
                                          škálovanie a posun, alebo zachovať naposledy nastavené
        :var self.__lock_view_check:      checkbox, ktorý zachytáva, či je táto možnosť zvolená alebo nie
        :var self.__3d_graph:             boolean premenná checboxu, ktorá označuje či sa má zobrazovať 2D alebo 3D graf
                                          možnosť je dostupná ak je počet neuronov na vrstve väčší alebo rovný ako 3
        :var self.__show_polygon:         boolean premenná checboxu, ktorá označuje či sa má zobrazovať mriežka, v 2D
                                          alebo 3D. Táto možnosť je dostupná ak je vstup do modleu tvoreným 2 alebo 3
                                          vstupmi.
        :var self.__dim_reduction_frame:  obaľovací widget, ktorý drží zoznam možných metód na redukciu priestoru.
                                          Sú tu zobrazované aj informácie z metódy prípadne zoznam nastaviteľných
                                          parametrov, potrebných pre použitie metódy
        :var self.__actual_used_label:    label, ktorý oboznámjue používateľa s práve použitou metódou redukcie
        :var self.__no_method_radio:      radio button, ktorý označuje že je zvolená metóda no_method
        :var self.__PCA_method_radio:     radio button, ktorý označuje že je zvolená metóda PCA
        :var self.__tSNE_method_radio:    radio button, ktorý označuje že je zvolená metóda t-SNE
        :var self.__PCA_info_frame:       obaľuje listboxy obashujúce informácie po použití metódy PCA
        :var self.__PC_explanation_frame: obsahuje listbox, v ktorom je vyjadrená koľko percent variability je
                                          vysvetelných jednotlivými komponentmi PCA
        :var self.__PC_explanation_lb:    listbox v ktorom je zobrazené aká variabilita je vyjadrená jednotlivými PC
        :var self.__PC_scores_frame:      obsahuje listbox, ktorý udáva ktoré neuróny majú najväčšiu váhu pri PCA
        :var self.__PC_scores_lb:         listbox v ktorom sú zoradené neuróny s najäčším vplyvom
        :var self.__tSNE_parameter_frame: obaľovací widget, obsahuje zoznam nastaviteľných parametrov, potrebných pre
                                          metódu t-SNE
        :var self.__tSNE_parameters_dict: slovník, ku v ktorom su k jednotlivým názvom parametrov priradené rewritable
                                          labels
        :var self.__apply_method_btn:     tlačidlo na použitie zvolenej metódy pomocou radio buttons

        Parmetre
        ----------------------------------------------------------------------------------------------------------------
        :param parent: nadradený tkinter Widget
        :param logicalLayer: odkaz na logickú vrstvu grafu
        """
        super().__init__(*args, **kwargs)
        # self.__graph_logic = logicalLayer
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setFont(font)
        self.__labels_entries_list = []
        self.__cords_entries_list = []
        self.__tSNE_parameters_dict = {}
        self.__changed_config = None
        self.__active_layer = None
        self.__layer_name_label = QLabel('Layer name')
        self.__bar_wrapper = QGroupBox('Layer Options')
        self.__choose_cords_gbox = QGroupBox('Choose Cords')
        self.__possible_cords_label = QLabel('Possible Cords')
        self.__choose_label_gbox = QGroupBox('Label names')
        self.__graph_view_gbox = QGroupBox('Graph view options')
        self.__color_labels_cb = QCheckBox('Color labels')
        self.__lock_view_cb = QCheckBox('Lock view')
        self.__3d_view_cb = QCheckBox('3D view')
        self.__show_polygon_cb = QCheckBox('Show polygon')
        self.__dim_reduction_gbox = QGroupBox('Dimension reduction')
        self.__actual_used_reduction_method = QLabel('Actual used: No method')
        self.__no_method_rb = QRadioButton('No method')
        self.__PCA_rb = QRadioButton('PCA')
        self.__t_SNE_rb = QRadioButton('t-SNE')
        self.__pca_info_gbox = QGroupBox('PCA information')
        self.__var_expl_lb = QListWidget()
        self.__load_scores_lb = QListWidget()
        self.__t_sne_parameter_gbox = QGroupBox('t-SNE parameters')
        self.__use_method_btn = QPushButton()
        self.__currently_used_method = 'No method'

        self.initialize_ui()

        # Obalovaci element. Ostáva stále zobrazený.

    def initialize_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(5, 0, 5, 5)
        self.setLayout(main_layout)
        options_groups_layout = QVBoxLayout()
        options_groups_layout.setAlignment(QtCore.Qt.AlignTop)
        options_groups_layout.setSpacing(0)
        options_groups_layout.setContentsMargins(0, 0, 0, 0)

        self.__layer_name_label.setMargin(10)

        layer_name_font = QtGui.QFont()
        layer_name_font.setPointSize(10)
        layer_name_font.setBold(True)
        self.__layer_name_label.setFont(layer_name_font)

        options_groups_layout.addWidget(self.__layer_name_label, alignment=QtCore.Qt.AlignCenter)

        self.__bar_wrapper.setLayout(options_groups_layout)
        main_layout.addWidget(self.__bar_wrapper)

        choose_cords_layout = QVBoxLayout()
        choose_cords_layout.setAlignment(QtCore.Qt.AlignTop)

        self.__choose_cords_gbox.setLayout(choose_cords_layout)

        options_groups_layout.addWidget(self.__choose_cords_gbox)

        options_group_title_font = QtGui.QFont()
        options_group_title_font.setPointSize(10)

        # self.__possible_cords_label.setFont(options_group_title_font)
        self.__possible_cords_label.setContentsMargins(0, 0, 0, 0)

        choose_cords_layout.addWidget(self.__possible_cords_label, alignment=QtCore.Qt.AlignHCenter)

        options_groups_layout.addWidget(self.__choose_label_gbox)
        choose_labels_layout = QVBoxLayout()
        self.__choose_label_gbox.setLayout(choose_labels_layout)
        choose_labels_layout.setAlignment(QtCore.Qt.AlignTop)

        label_entry_names = ['X axis label:', 'Y axis label:', 'Z axis label:']
        label_entry_values = ['Label X', 'Label Y', 'Label Z']
        cords_entry_names = ['Axis X:', 'Axis Y:', 'Axis Z:']
        for i in range(3):
            rw_label = RewritableLabel(i, cords_entry_names[i], str('-'), self.validate_cord_entry)
            choose_cords_layout.addWidget(rw_label, alignment=QtCore.Qt.AlignLeft)
            self.__cords_entries_list.append(rw_label)
            rw_label = RewritableLabel(i, label_entry_names[i], label_entry_values[i], self.validate_label_entry)
            choose_labels_layout.addWidget(rw_label, alignment=QtCore.Qt.AlignLeft)
            self.__labels_entries_list.append(rw_label)

        options_groups_layout.addWidget(self.__graph_view_gbox)

        view_options_layout = QVBoxLayout()
        self.__graph_view_gbox.setLayout(view_options_layout)
        view_options_layout.setAlignment(QtCore.Qt.AlignTop)

        self.__color_labels_cb.toggled.connect(self.on_color_label_check)
        self.__lock_view_cb.toggled.connect(self.on_lock_view_check)
        self.__3d_view_cb.toggled.connect(self.on_3d_graph_check)
        self.__show_polygon_cb.toggled.connect(self.on_show_polygon_check)

        view_options_layout.addWidget(self.__color_labels_cb)
        view_options_layout.addWidget(self.__lock_view_cb)
        view_options_layout.addWidget(self.__3d_view_cb)
        view_options_layout.addWidget(self.__show_polygon_cb)

        options_groups_layout.addWidget(self.__dim_reduction_gbox)
        dim_reduction_layout = QVBoxLayout()
        dim_reduction_layout.setContentsMargins(0, 0, 0, 0)
        dim_reduction_layout.setAlignment(QtCore.Qt.AlignTop)
        self.__dim_reduction_gbox.setLayout(dim_reduction_layout)

        self.__actual_used_reduction_method.setMargin(10)
        dim_reduction_layout.addWidget(self.__actual_used_reduction_method,
                                       alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        radio_button_group = QButtonGroup()
        radio_button_layout = QHBoxLayout()
        radio_button_layout.setContentsMargins(5, 0, 5, 5)
        radio_button_layout.setAlignment(QtCore.Qt.AlignLeft)
        dim_reduction_layout.addLayout(radio_button_layout)

        self.__no_method_rb.toggled.connect(self.on_method_change)
        self.__PCA_rb.toggled.connect(self.on_method_change)
        self.__t_SNE_rb.toggled.connect(self.on_method_change)

        self.__no_method_rb.setChecked(True)
        radio_button_group.addButton(self.__no_method_rb)
        radio_button_layout.addWidget(self.__no_method_rb)

        radio_button_group.addButton(self.__PCA_rb)
        radio_button_layout.addWidget(self.__PCA_rb)

        radio_button_group.addButton(self.__t_SNE_rb)
        radio_button_layout.addWidget(self.__t_SNE_rb)

        self.__pca_info_gbox.setMaximumHeight(200)
        dim_reduction_layout.addWidget(self.__pca_info_gbox, alignment=QtCore.Qt.AlignTop)

        pca_info_layout = QHBoxLayout()
        pca_info_layout.setAlignment(QtCore.Qt.AlignTop)
        pca_info_layout.setContentsMargins(0, 10, 0, 0)
        self.__pca_info_gbox.setLayout(pca_info_layout)

        var_expl_gbox = QGroupBox('PC variance explanation')

        pca_info_layout.addWidget(var_expl_gbox)
        #
        layout_var_expl = QVBoxLayout()
        layout_var_expl.setAlignment(QtCore.Qt.AlignTop)
        layout_var_expl.setContentsMargins(0, 0, 0, 0)
        var_expl_gbox.setLayout(layout_var_expl)
        self.__var_expl_lb.setMaximumWidth(150)
        layout_var_expl.addWidget(self.__var_expl_lb)

        loading_score_gbox = QGroupBox('Loading scores')
        pca_info_layout.addWidget(loading_score_gbox)

        loading_score_layout = QVBoxLayout()
        loading_score_layout.setAlignment(QtCore.Qt.AlignTop)
        loading_score_gbox.setLayout(loading_score_layout)
        loading_score_layout.setContentsMargins(0, 0, 0, 0)
        self.__load_scores_lb.setMaximumWidth(150)
        loading_score_layout.addWidget(self.__load_scores_lb)

        dim_reduction_layout.addWidget(self.__t_sne_parameter_gbox, alignment=QtCore.Qt.AlignTop)
        t_sne_par_layout = QVBoxLayout()
        t_sne_par_layout.setAlignment(QtCore.Qt.AlignTop)
        self.__t_sne_parameter_gbox.setLayout(t_sne_par_layout)

        t_sne_parameter_id_list = ['n_components', 'perplexity', 'early_exaggeration', 'learning_rate', 'n_iter']
        t_sne_parameter_label = ['Number of components:', 'Perplexity:', 'Early exaggeration:', 'Learning rate:',
                                 'Number of iteration:']

        for i in range(len(t_sne_parameter_id_list)):
            t_sne_parameter_rw = RewritableLabel(t_sne_parameter_id_list[i], t_sne_parameter_label[i], '-',
                                                 self.validate_t_sne_entry)
            t_sne_par_layout.addWidget(t_sne_parameter_rw, alignment=QtCore.Qt.AlignLeft)
            self.__tSNE_parameters_dict[t_sne_parameter_id_list[i]] = t_sne_parameter_rw

        self.__use_method_btn.setText('Use method')
        dim_reduction_layout.addWidget(self.__use_method_btn)
        self.__use_method_btn.clicked.connect(self.use_selected_method)

    def initialize(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vyčistenie atribútov a skrytie celého options baru.
        """
        self.__changed_config = None
        self.__active_layer = None
        self.hide_option_bar()

    def initialize_with_layer_config(self, neural_layer, config):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie aktuálnej aktívnej vrstvy a configu tejto vrstvy.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param neural_layer: odkaz na aktívnu vrstvu, pre ktorú sú menené nastavenia a ktoré budú následne na túto
                             vrstvu použité
        :param config: odkaz na config aktuálne zobrazovanej a upravovanej vrstvy
        """
        self.__active_layer = neural_layer
        self.__changed_config = config
        self.update_selected_config()

    def initialize_label_options(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie jednotlivých vstupov pre časť s názvami os grafov na základe načítaného configu.
        """
        number_of_possible_dim = self.__changed_config['max_visible_dim']
        for i in range(number_of_possible_dim):
            label_entry = self.__labels_entries_list[i]
            label_entry.set_variable_label(self.__changed_config['axis_labels'][i])
            label_entry.show()

    def initialize_view_options(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie hodnôt jednotlivých vstupov pre časť s možnosťami zobrazovania grafu na základe načítaného configu.
        """
        self.__color_labels_cb.setChecked(self.__changed_config['color_labels'])
        self.__lock_view_cb.setChecked(self.__changed_config['locked_view'])

        number_of_possible_dim = self.__changed_config['max_visible_dim']

        if number_of_possible_dim >= 3:
            self.__3d_view_cb.setChecked(self.__changed_config['draw_3d'])
            self.__3d_view_cb.show()

        if self.__changed_config['possible_polygon']:
            self.__show_polygon_cb.setChecked(self.__changed_config['show_polygon'])
            self.__show_polygon_cb.show()
        
        if self.__active_layer.possible_color_labels:
            self.__color_labels_cb.setChecked(self.__changed_config['color_labels'])
            self.__color_labels_cb.show()

    def initialize_dimension_reduction_options(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie hodnôt jednotlivých vstuov pre časť s možnosťami pre redukciu priestoru.
        """
        self.set_actual_method_lable(self.__currently_used_method)
        config_selected_method = self.__changed_config['config_selected_method']

        self.__no_method_rb.setChecked(False)
        self.__PCA_rb.setChecked(False)
        self.__t_SNE_rb.setChecked(False)

        if config_selected_method == 'No method':
            self.__no_method_rb.setChecked(True)
        elif config_selected_method == 'PCA':
            self.__PCA_rb.setChecked(True)
        elif config_selected_method == 't-SNE':
            self.__t_SNE_rb.setChecked(True)

        if self.__currently_used_method == 'PCA':
            self.update_PCA_information()

        self.initialize_t_sne_parameters()
        self.on_method_change()

    def initialize_t_sne_parameters(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Predvyplnenie parametrov pre metódu t-SNE.
        """
        t_sne_config = self.__changed_config['t_SNE_config']
        actual_used_config = t_sne_config['used_config']
        options_config = t_sne_config['options_config']
        number_of_components = t_sne_config['parameter_borders']['n_components'][2]
        self.__tSNE_parameters_dict['n_components'].set_label_name(
            f'Number of components (max {number_of_components}):')
        for key in self.__tSNE_parameters_dict:
            rewritable_label = self.__tSNE_parameters_dict[key]
            rewritable_label.set_variable_label(options_config[key])

            # Ak sa aktuálne používaná hodnota parametra nerovná naposledy nastavenej hodnote parametra je tento
            # parameter označený ako zmenený no ešte nepoužitý.
            if actual_used_config[key] == options_config[key]:
                rewritable_label.set_mark_changed(False)
            else:
                rewritable_label.set_mark_changed(True)

    def use_selected_method(self):
        if self.__active_layer is not None:
            method = self.get_checked_method()
            need_recalculation = False
            if method == 't-SNE':
                need_recalculation = self.apply_t_SNE_options_if_changed()

            if method != self.__changed_config['used_method']:
                if method == 'PCA':
                    self.__changed_config['PCA_config']['displayed_cords'] = list(range(min(self.__changed_config['output_dimension'],
                                                                                            self.__changed_config['number_of_samples'],
                                                                                            3)))

                need_recalculation = True
                self.__changed_config['used_method'] = self.__currently_used_method = method
            if need_recalculation:
                self.__changed_config['apply_changes'] = True
                self.__active_layer.use_config()
                self.set_actual_method_lable(method)

                # Hide all informations
                self.hide_all_methods_information()
                if method == 'PCA':
                    self.update_PCA_information()
                    self.__pca_info_gbox.show()
                elif method == 't-SNE':
                    self.__t_sne_parameter_gbox.show()

                self.set_actual_method_lable(method)
            self.set_cords_entries_according_chosen_method()

    def update_active_options_layer(self, start_layer=-1):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zobrazenie aktuálnych informácií pre aktívnu vrstvu, ktorej možnosti sú zobrazované.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param start_layer: poradové číslo najnižšej vrstvy, pri ktorej došlo ku zmene, ak je číslo menšie ako poradové
                            číslo aktívnej vrstvy, ktorej možnosti sú zobrazované, sú tieto možnosti aktualizované,
                            pretože mohlo dôjsť k zmenám v možných parametroch, prípadne k zmene hodnôt v rámci PCA.
        """
        if self.__active_layer is not None and self.__changed_config is not None:
            actual_method = self.__changed_config['used_method']
            if actual_method == 'PCA' and start_layer < self.__active_layer.layer_number:
                self.update_PCA_information()

    def update_selected_config(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie jednotlivých možností na základe configu aktívnej vrstvy.
        """
        if self.__active_layer is not None and self.__changed_config is not None:
            self.__currently_used_method = self.__changed_config['used_method']
            self.hide_option_bar()
            self.__bar_wrapper.show()
            self.__layer_name_label.setText(str(self.__changed_config['layer_name']))

            self.set_cords_entries_according_chosen_method()
            self.initialize_label_options()
            self.initialize_view_options()
            self.initialize_dimension_reduction_options()

    def update_PCA_information(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vypísanie aktuálnych informácii o PCA.
        """
        variance_series = self.__changed_config['PCA_config']['percentage_variance']
        if variance_series is not None:
            self.__var_expl_lb.clear()
            pc_labels = variance_series.index
            for i, label in enumerate(pc_labels):
                self.__var_expl_lb.insertItem(i, '{}: {:.2f}%'.format(label, round(variance_series[label], 2)))
            loading_scores = self.__changed_config['PCA_config']['largest_influence']

            self.__load_scores_lb.clear()
            # Zoradenie významností jednotlivých neurónov pri PCA
            sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
            sorted_indexes = sorted_loading_scores.index.values
            for i, label in enumerate(sorted_indexes):
                self.__load_scores_lb.insertItem(i, '{}: {:.4f}'.format(label, round(loading_scores[label], 4)))

    def update_active_options_layer(self, start_layer=-1):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zobrazenie aktuálnych informácií pre aktívnu vrstvu, ktorej možnosti sú zobrazované.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param start_layer: poradové číslo najnižšej vrstvy, pri ktorej došlo ku zmene, ak je číslo menšie ako poradové
                            číslo aktívnej vrstvy, ktorej možnosti sú zobrazované, sú tieto možnosti aktualizované,
                            pretože mohlo dôjsť k zmenám v možných parametroch, prípadne k zmene hodnôt v rámci PCA.
        """
        if self.__active_layer is not None and self.__changed_config is not None:
            actual_method = self.__changed_config['used_method']
            if actual_method == 'PCA' and start_layer < self.__active_layer.layer_number:
                self.update_PCA_information()

    def apply_t_SNE_options_if_changed(self):
        changed = False
        if self.__changed_config is not None:
            t_sne_config = self.__changed_config['t_SNE_config']
            used_config = t_sne_config['used_config']
            options_config = t_sne_config['options_config']
            for key in used_config:
                if used_config[key] != options_config[key]:
                    changed = True
                    used_config[key] = options_config[key]
            if changed:
                t_sne_config['displayed_cords'] = list(range(used_config['n_components']))
                self.set_entries_not_marked(self.__tSNE_parameters_dict.values())
        return changed

    def set_cords_entries_according_chosen_method(self):
        if self.__currently_used_method == 'No method':
            if self.__changed_config['has_feature_maps']:
                if self.__currently_used_method == 'No method':
                    output_shape = self.__changed_config['output_shape']
                    entry_names = ['Axis X:', 'Axis Y:', 'Axis Z:']
                    cords_label_text = 'Possible cords: {}x{}'.format(output_shape[-3], output_shape[-2])
                    tmp_displayed_cords_tuples = list(
                        zip(self.__changed_config['no_method_config']['displayed_cords'][0],
                            self.__changed_config['no_method_config']['displayed_cords'][1]))
                    displayed_cords = []
                    for x, y in tmp_displayed_cords_tuples:
                        displayed_cords.append(f'{x}, {y}')
                    possible_cords = self.__changed_config['max_visible_dim']
            else:
                entry_names = ['Axis X:', 'Axis Y:', 'Axis Z:']
                cords_label_text = 'Possible cords: 0-{}'.format(self.__changed_config['output_dimension'] - 1)
                displayed_cords = self.__changed_config['no_method_config']['displayed_cords']
                possible_cords = self.__changed_config['max_visible_dim']
        elif self.__currently_used_method == 'PCA':
            entry_names = ['PC axis X:', 'PC axis Y:', 'PC axis Z:']
            number_of_pcs = min(self.__changed_config['output_dimension'],
                                self.__changed_config['number_of_samples'])
            if number_of_pcs == 0:
                cords_label_text = 'No possible PCs:'
            else:
                cords_label_text = 'Possible PCs: 1-{}'.format(number_of_pcs)
            possible_cords = min(number_of_pcs, self.__changed_config['max_visible_dim'])
            displayed_cords = self.__changed_config['PCA_config']['displayed_cords'].copy()
            displayed_cords = np.array(displayed_cords) + 1
        elif self.__currently_used_method == 't-SNE':
            entry_names = ['t-SNE X:', 't-SNE Y:', 't-SNE Z:']
            possible_cords = self.__changed_config['t_SNE_config']['used_config']['n_components']
            cords_label_text = 'Possible t-SNE components: 0-{}'.format(possible_cords - 1)
            displayed_cords = self.__changed_config['t_SNE_config']['displayed_cords'].copy()

        self.set_cords_entries(entry_names, cords_label_text, displayed_cords, possible_cords)

    def set_cords_entries(self, entry_name, cords_label_text, displayed_cords, possible_cords):
        self.hide_choose_cords_entries()
        self.__possible_cords_label.setText(str(cords_label_text))
        for i in range(possible_cords):
            cord_entry_rewritable_label = self.__cords_entries_list[i]
            cord_entry_rewritable_label.set_label_name(entry_name[i])
            cord_entry_rewritable_label.set_variable_label(displayed_cords[i])
            cord_entry_rewritable_label.show()

    def set_actual_method_lable(self, method_name):
        self.__actual_used_reduction_method.setText(str(f'Actual used: {method_name}'))

    def set_entries_not_marked(self, entries_list):
        for entry in entries_list:
            entry.set_mark_changed(False)

    def hide_option_bar(self):
        self.__bar_wrapper.hide()
        self.hide_option_bar_items()

    def hide_option_bar_items(self):
        self.hide_choose_cords_entries()
        self.hide_label_entries()
        self.hide_graph_view_options()
        self.hide_dimension_reduction_options()

    def hide_choose_cords_entries(self):
        for i in range(3):
            self.__cords_entries_list[i].hide()

    def hide_label_entries(self):
        for i in range(3):
            self.__labels_entries_list[i].hide()

    def hide_graph_view_options(self):
        self.__color_labels_cb.hide()
        self.__3d_view_cb.hide()
        self.__show_polygon_cb.hide()

    def hide_dimension_reduction_options(self):
        self.hide_all_methods_information()

    def hide_all_methods_information(self):
        self.__pca_info_gbox.hide()
        self.__t_sne_parameter_gbox.hide()

    def on_method_change(self):
        print('zmena metody')
        if self.__changed_config:
            method = self.get_checked_method()
            self.__changed_config['config_selected_method'] = method
            if method == 'PCA':
                if self.__changed_config['used_method'] == method:
                    self.__pca_info_gbox.show()
                self.__t_sne_parameter_gbox.hide()
            elif method == 't-SNE':
                self.__pca_info_gbox.hide()
                self.__t_sne_parameter_gbox.show()
            else:
                self.__pca_info_gbox.hide()
                self.__t_sne_parameter_gbox.hide()

    def on_color_label_check(self):
        if self.__changed_config:
            self.__changed_config['color_labels'] = self.__color_labels_cb.isChecked()
            self.__active_layer.use_config()

    def on_lock_view_check(self):
        if self.__changed_config:
            self.__changed_config['locked_view'] = self.__lock_view_cb.isChecked()
            self.__active_layer.use_config()

    def on_show_polygon_check(self):
        if self.__changed_config:
            self.__changed_config[
                'show_polygon'] = self.__active_layer.calculate_polygon = self.__show_polygon_cb.isChecked()
            if self.__changed_config['show_polygon']:
                self.__active_layer.set_polygon_cords()
            self.__active_layer.use_config()

    def on_3d_graph_check(self):
        if self.__changed_config:
            self.__changed_config['draw_3d'] = self.__3d_view_cb.isChecked()
            self.__active_layer.use_config()

    def validate_label_entry(self, id, value):
        self.__labels_entries_list[id].set_variable_label(value)
        self.__labels_entries_list[id].show_variable_label()
        self.__changed_config['axis_labels'][id] = value
        self.__active_layer.use_config()

    def validate_cord_entry(self, id, value):
        bottom_border = 0
        top_border = 0
        changed_cords = None
        if self.__currently_used_method == 'No method':
            if self.__changed_config['has_feature_maps']:
                if self.__currently_used_method == 'No method':
                    changed_cords = self.__changed_config['no_method_config']['displayed_cords']
                    output_shape = self.__changed_config['output_shape']
                    borders = [output_shape[-3] - 1, output_shape[-2] - 1]
                    return_val = False
                    correct_input = []
                    output_text = ['', '']
                    entry_input = value.split(',')
                    if len(entry_input) == 2:
                        for i, number in enumerate(entry_input):
                            try:
                                number_int = int(number)
                                if 0 <= number_int <= borders[i]:
                                    correct_input.append(int(number))
                                    output_text[i] = int(number)
                                else:
                                    output_text[i] = 'err'
                            except ValueError:
                                output_text[i] = 'err'

                        output_msg = f'{output_text[0]}, {output_text[1]}'
                        if len(correct_input) == 2:
                            changed_cords[0][id] = correct_input[0]
                            changed_cords[1][id] = correct_input[1]
                            self.__cords_entries_list[id].set_entry_text(output_msg)
                            return_val = True
                        else:
                            return_val = False
                    else:
                        output_msg = 'err'
                        return_val = False

                    self.__cords_entries_list[id].set_variable_label(output_msg)
                    self.__cords_entries_list[id].show_variable_label()
                    self.__changed_config['cords_changed'] = True
                    if return_val:
                        self.__active_layer.use_config()
                    return return_val
            else:
                bottom_border = 0
                top_border = self.__changed_config['output_dimension']
                changed_cords = self.__changed_config['no_method_config']['displayed_cords']
                new_value = int(value)
        elif self.__currently_used_method == 'PCA':
            bottom_border = 1
            top_border = min(self.__changed_config['output_dimension'],
                             self.__changed_config['number_of_samples']) + 1
            changed_cords = self.__changed_config['PCA_config']['displayed_cords']
            new_value = int(value) - 1
        elif self.__currently_used_method == 't-SNE':
            bottom_border = 0
            top_border = self.__changed_config['t_SNE_config']['used_config']['n_components']
            changed_cords = self.__changed_config['t_SNE_config']['displayed_cords']
            new_value = int(value)
        try:
            if not (bottom_border <= int(value) < top_border):
                self.__cords_entries_list[id].set_entry_text('err')
                return False

            self.__cords_entries_list[id].set_variable_label(value)
            self.__cords_entries_list[id].show_variable_label()
            changed_cords[id] = int(new_value)
            self.__changed_config['cords_changed'] = True
            self.__active_layer.use_config()
            return True
        except ValueError:
            self.__cords_entries_list[id].set_entry_text('err')
            return False

    def validate_t_sne_entry(self, id, value):
        try:
            if self.__changed_config is not None:
                test_tuple = self.__changed_config['t_SNE_config']['parameter_borders'][id]
                if not (test_tuple[0] <= test_tuple[1](value) <= test_tuple[2]):
                    self.__tSNE_parameters_dict[id].set_entry_text('err')
                    return False
                parameter_label = self.__tSNE_parameters_dict[id]
                parameter_label.set_variable_label(value)
                parameter_label.show_variable_label()
                self.__changed_config['t_SNE_config']['options_config'][id] = test_tuple[1](value)
                if self.__changed_config['t_SNE_config']['options_config'][id] == \
                        self.__changed_config['t_SNE_config']['used_config'][id]:
                    parameter_label.set_mark_changed(False)
                else:
                    parameter_label.set_mark_changed(True)
                return True
            else:
                return False
        except ValueError:
            self.__tSNE_parameters_dict[id].set_entry_text('err')
            return False

    def get_checked_method(self):
        if self.__no_method_rb.isChecked():
            return 'No method'
        elif self.__PCA_rb.isChecked():
            return 'PCA'
        elif self.__t_SNE_rb.isChecked():
            return 't-SNE'
        return None

    @property
    def active_layer(self):
        return self.__active_layer


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


sys.excepthook = except_hook
app = QApplication(sys.argv)
window = VisualizationApp()
window.showMaximized()

app.exec_()
