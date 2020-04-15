import sys

from LogicComponents import *


class VisualizationApp(QMainWindow):
    def __init__(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Spúšťacia trieda. Je obalom celeje aplikácie. Vytvorí programové okno, na celú obrazovku.

        """
        # Konštruktor základnej triedy.
        # super(VisualizationApp, self).__init__(*args, **kwargs)
        super(VisualizationApp, self).__init__(*args, **kwargs)

        self.setWindowTitle('NN visualization tool')
        self.setContentsMargins(0, 0, 0, 0)
        graphPage = GraphPage(self)
        self.setCentralWidget(graphPage)


class GraphPage(QWidget):
    def __init__(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Podstránka, na ktorej je zobrazovaný rámec s grafmi, váhami a okno s detailmi vrstvy.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__file_path:   Cesta k súboru pre lepší konfort pri načítaní a ukladaní.
        :var self.__file_name:   Meno súboru pre lepší konfort pri ukladaní.
        :var self.__keras_model: Načítaný model, jeho referencia je zaslaná do logic layer, kde sa menia váhy.
                                 Používa sa aj pri ukladaní.
        :var self.__logic_layer: Referencia na logiku aplikácie.
        :var self.__mg_frame:    Referencia na hlavné okno, ktoré bude zobrazovať grafy vrstiev a ovládače váh.
        :var self.__load_ilabel: Zobrazuje informácie o načítaní vstupov.
        :var self.__save_ilabel: Zobrazuje informáciu o tom, či je možné uložiť model.

        """
        super(GraphPage, self).__init__(*args, **kwargs)

        # Inicializácia atribútov.
        self.__file_path = '.'
        self.__file_name = ''
        self.__keras_model = None
        self.__logic_layer = None
        self.__mg_frame = MainGraphFrame(self)
        self.__load_ilabel = QLabel()
        self.__save_ilabel = QLabel()

        # Grafická inicializácia.
        self.initialize_ui()

        # Vytvorenie logickej vrstvy aplikácie.
        self.__logic_layer = GraphLogicLayer(self.__mg_frame)

        # self.open_model('./modelik.h5')
        self.open_model('./cnn_mnist.h5')
        # self.open_model('./small_1channel.h5')
        # self.open_model('./small_cnn.h5')
        # self.load_points('./2d_input.txt')

    def initialize_ui(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializácia, rozloženie a zobrazenie grafických komponentov pre stránku.
        """
        # Inicializácia základného rozmiestnenia okna. Na vrchu sa nachádza wrapper tlačidiel. Pod ním priestor na hlavné
        # okno, v bude zobrazovať ovládací panel a scrollovacie okno, obsahujúce reprezentácie výstupu vrstiev.
        vertical_layout = QVBoxLayout(self)
        vertical_layout.setContentsMargins(0, 0, 0, 0)
        vertical_layout.setSpacing(0)

        # Wrapper na skupiny tlačidiel. Je umiestený navrchu okna.
        button_wrapper = QWidget(self)
        button_wrapper.setFixedHeight(50)
        vertical_layout.addWidget(button_wrapper)

        # Rozloženie v rámci wrappera tlačidiel. Ide o horizontálne rozmiestnenie.
        button_layout = QHBoxLayout(button_wrapper)
        button_layout.setSpacing(0)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # Wrapper pre skupinu tlačidiel, open model, load inputs, load label a grafického prvku typu label, ktorý je
        # použitý na zobrazovanie informácií ohľadom chýb pri otváraní súborov alebo pri ich prideľovaní.
        group_wrapper_open_load = QWidget(self)

        # Rozmiestnenie v rámci samotného wrappera skupiny tlačidiel open model, load inputs a load labels spolu s label
        # elementom určeným pre zobrazovanie informácií.
        open_load_layout = QHBoxLayout(group_wrapper_open_load)
        open_load_layout.setSpacing(5)
        open_load_layout.setContentsMargins(0, 0, 0, 0)

        # Obaľovací grafický element pre skupinu tvorenú tlačidlom pre uloženie modelu a label elementom určeným na zob-
        # razovanie informácií ohľadom ukladania modelu.
        group_wrapper_save = QWidget(self)

        # Inicializácia rozmiestnenia skupiny obsahujúcej tlačidlo na ukladanie modelu a label zobrazujúci informácie.
        save_layout = QHBoxLayout(group_wrapper_save)
        save_layout.setSpacing(5)
        save_layout.setContentsMargins(0, 0, 0, 0)

        # Nastavenie textu, ktorý sa zobrazí pri spustení aplikácie, keď ešte nie je načítaný žiadny model.
        self.__load_ilabel.setText("<font color='orange'>Open Keras model.</font>")

        # Nastavenie fontu pre informačné labely. Je im nastavená veľkosť a taktiež bold.
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.__load_ilabel.setFont(font)
        self.__save_ilabel.setFont(font)

        # Inicializácia tlačidla pre otvorenie modelu. Nastavenie textu zobrazovaného na tlačidle, nastavenie funkcie,
        # ktorá sa má vykonať po stlačení tlačidla a pridanie tlačidla do rozmiestnenia tlačidiel.
        open_btn = QPushButton('Open model', group_wrapper_open_load)
        open_btn.clicked.connect(self.try_open_model)
        open_load_layout.addWidget(open_btn)

        # Inicializácia tlačidla pre načítanie vstupu. Natavenie textu zobrazovaného na tlačidle, nastavenie funkcie,
        # ktorá sa vykoná po kliknutí na tlačidlo. Pridanie tlačidla do rozmiestenia tlačidiel v rámci wrappera.
        load_inputs_btn = QPushButton('Load inputs', group_wrapper_open_load)
        load_inputs_btn.clicked.connect(self.try_load_points)
        open_load_layout.addWidget(load_inputs_btn)

        # Inicializácia tlačidla pre načítanie labelov načítaných bodov. Natavenie textu zobrazovaného na tlačidle,
        # nastavenie funkcie, ktorá sa vykoná po kliknutí na tlačidlo.
        # Pridanie tlačidla do rozmiestenia tlačidiel v rámci wrappera.
        load_labels_btn = QPushButton('Load labels', group_wrapper_open_load)
        load_labels_btn.clicked.connect(self.try_load_labels)
        open_load_layout.addWidget(load_labels_btn)

        # Priadanie label elementu, zobrazujúceho informácie do rozmiestnenia v rámci skupiny tlačidiel pre otvorenie
        # modelu, načítanie vstupov a načítania labels.
        open_load_layout.addWidget(self.__load_ilabel, alignment=QtCore.Qt.AlignRight)

        # Inicializácia tlačidla pre uloženie modelu. Natavenie textu zobrazovaného na tlačidle,
        # nastavenie funkcie, ktorá sa vykoná po kliknutí na tlačidlo.
        # Pridanie tlačidla do rozmiestenia tlačidiel v rámci wrappera.
        save_btn = QPushButton('Save model', group_wrapper_save)
        save_btn.clicked.connect(self.save_model)
        save_layout.addWidget(save_btn)

        # Pridanie informačného label elementu o ukladaní modelu do rozmiestnenia v rámci skupiny obsahujúcej tlačidlo
        # na uloženie modelu.
        save_layout.addWidget(self.__save_ilabel)

        # Pridanie wrappera skupiny načítacích elementov do rozmiestnenia. Nachádza sa na ľavej strane a je taktiež
        # zarovnaný naľavo.
        button_layout.addWidget(group_wrapper_open_load, alignment=QtCore.Qt.AlignLeft)

        # Pridanie wrappera obsahujúceho skupinu ukládacích elementov. Je vložený na pravú stranu a je taktiež zarovnaný
        # napravo.
        button_layout.addWidget(group_wrapper_save, alignment=QtCore.Qt.AlignRight)

        # Pridanie hlavného okna do rozmiestnenia stránky.
        vertical_layout.addWidget(self.__mg_frame)

    def try_open_model(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zobrazí dialógové okno na výber Keras modelu. Ak bude zadaná nejaká cesta k tomuto súboru, bude cesta uložená a
        model otvorený.
        """
        # Metóda dialógového okna vracia list vo formáte [cesta ku súboru, zvolený typ súboru]. Pomocou indexu 0 teda
        # získame cestu k zvolenému súboru. Ak je cesta k súboru prázdna, program nevykoná žiadnu akciu. Ak bola
        # zadaná nejaká cesta k súboru, pokračuje sa do metódy open_model.
        file_path = QFileDialog.getOpenFileName(self, 'Open Keras model', self.__file_path, 'Keras model (*.h5)')[0]
        if file_path != '':
            self.open_model(file_path)

    def open_model(self, file_path):
        """
        Popis:
        ----------------------------------------------------------------------------------------------------------------
        Načítanie súboru na základe cesty. Rozdelenie zadanej cesty k súboru na absolútnu cestu a názov súboru.
        Inicializácia logickej vrstvy, načítaným modelom.
        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param file_path: aboslutná cesta k súboru.
        """
        # Cesta k súboru sa rozdelí na adresu smerujúcu do priečinka obsahujúcu súbor a názov súboru. Tie sú priradené
        # do premenných, na základe ktorých sa pri ďalšom otváraní modelu, alebo načítaní vstupov dialogové okno otvorí
        # s otvoreným priečinkom, z ktorého bol model načítaný.
        self.__file_path, self.__file_name = ntpath.split(file_path)

        # Keras funkcia, ktorá načíta model na zákalde cestky k súboru.
        self.__keras_model = keras.models.load_model(file_path)

        # Inicializácia logickej vrstvy na základe načítaného modelu.
        self.__logic_layer.initialize(self.__keras_model)

        # Ak bolo zobrazené nejaké upozornenie, alebo chyba ohľadom načítania modelu, bude táto skrytá.
        self.__load_ilabel.hide()
        self.__save_ilabel.hide()

    def try_load_labels(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zobrazí dialógové okno na výber súboru obsahujucého vstupy. Súbory môžu byť typu .txt alebo .csv. Ak bola
        zvolená nejaká cesta ku súboru, pokraćuje sa načítaním vstupu. Ak došlo k chybe, bude zobrazená vedľa tlačidla.
        """
        # Načítanie labelov k jednotlivým bodom, môže prebehnúť len ak bol už nejaký model neurónovej siete načítaný.
        # Ak žiaden z modelov nebol načítaný zobrazí sa varovanie, že je potrebné najskôr nejaký model načítať.
        if self.__keras_model is not None:
            # Z dialógového okna sa snažíme dostať cestu k súboru. Ak žiadna cesta k súboru nebola zadaná, nevykoná sa
            # žiadna akcia. Ak bola zadaná nejaká cesta k súboru, pokračuje sa ďalej k načítaniu labels v rámci logickej
            # vrstvy.
            file_path = \
                QFileDialog.getOpenFileName(self, 'Load labels', self.__file_path, 'Text or CSV file (*.txt *.csv)')[0]
            if file_path != '':
                # Načítaná cesta k súboru je zaslaná do metódy logickej vrstvy na načítanie súboru obsahujúceho labels.
                # Ak je návratová hodnota tejto metódy None, znamená to, že načítavanie prebehlo úspešne. Ak je hodnta
                # iná, znamená to, že došlo k udalosti, o ktorej by mal byť používateľ informovaný.
                error_message = self.__logic_layer.load_labels(file_path)
                if error_message is not None:
                    # Vypíše výstupnú hodnotu z metódy na načítanie labels pre body.
                    self.__load_ilabel.setText("<font color='red'>{}</font>".format(error_message))
                    self.__load_ilabel.show()
                else:
                    # Ak načítanie prebehlo úspešne, budú prípadné predchádzajúce informačné hlášky skryté.
                    self.__load_ilabel.hide()
        else:
            self.__load_ilabel.setText("<font color='red'>You have to load model first!</font>")

    def try_load_points(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Získanie adresy k súboru s hodnotami bodov. Ak nie je načítaný model, zobrazí sa informácia o chybe.
        """
        # Testovanie či je načítaný model neurónvej siete. V prípade, že model nie je načítaný je zobrazená hláška,
        # o tom, že pred načítaním bodov je potrebné načítať nejaký model.
        if self.__keras_model is not None:

            # Podľa toho aký je vstup do neurónovej siete je možné rozlíšiť, či sa jedná o klasickú husto prepojenú
            # neurónovú sieť alebo či ide o konvolučnú neurónovú sieť.
            input_shape = self.__keras_model.input_shape
            error_message = None
            file_path = ''

            # Ak vstupný rozmer neurónovej siete obsahuje viac ako dva prvky (prvý je väčšinou None no môže to byť aj
            # číslo, ktoré označuje veľkosť dávky), ide o konvolúčnu neurónovú sieť.

            if len(input_shape) > 2:
                # Vytvorenie messageboxu, ktorý slúži k získaniu spôsobu, akým chce používateľ načítať obrázky. Na výber
                # má z dvoch možností a to, Folder a Separately. Pomocou možnosti folder, sa objaví dialogové okno,
                # pomocou ktorého zvolí priečinok, v ktorom sú po zložkách rozdelené obrázky. Názvy zložiek
                # predstavujú labels, ktoré budú priradené obrázkom, ktoré z daného priečinku boli
                # načítané. Pomocou možnosti Separately si obrázky ktoré chce načítať vyberie
                # manuálne. Pri tomto spôsobe sa obrázkom nepriradí žiaden label.
                msgBox = CustomMessageBox('Load input', 'How would you like to load inputs?',
                                          'Folder', 'Separately', 'Cancel',
                                          'Folder: folder with labeled folders of images (assign labels)\n' +
                                          'Separately: manually image choosing')
                msgBox.exec_()

                # Tento blok kódu zisťuje, ktoré tlačidlo bolo zvolené a na základe tohto zistenia ponúkne používateľovi
                # dialógové okno, ktoré umožní vybrať buď to zložku alebo samostatné súbory. Možnosť folder je
                # reprezentovaná číslom 0. Možnosť 1 predstavuje zvolenie možnosti Separately. V ostatných
                # prípadoch bola zvolená možnosť Cancel, alebo bolo dialogové okno zatvorené.
                if msgBox.result() == 0:

                    # Ak si používateľ zvolil možnosť folder, pokačuje sa touto vetvou. Objaví sa dialogové okno nasme-
                    # rované do priečinka, z ktorého bol načítaný keras model. Používateľ si môže zvoliť priečinok,
                    # v ktorom sa nachádazajú priečniky obsahujúce obrázky podľa labelu. Dialogové okno vráti
                    # cestu k priečinku.
                    file_path = QFileDialog.getExistingDirectory(self, 'Choose folder', self.__file_path)

                    # Ak nebol žiaden priečinok zvolený program nevykoná žiadnu akciu. Ak bola zvolená cesta do priečin-
                    # ku, bude zavolaná metóda logickej vrstvy spolu, ktorá preberá ako argument zadanú cestu.
                    if file_path != "":
                        # Ak pri načítavaní bodov došlo chyba, je táto informácia vrátená ako výstupná hodnota. Táto
                        # hodnota sa následne zobrazí pomocou informačného elementu.
                        error_message = self.__logic_layer.handle_images_input(file_path)
                elif msgBox.result() == 1:

                    # Ak si používateľ zvolí možnosť Separately, pokračuje sa touto vetvou. Objaví sa dialogovéh okno,
                    # pomocou ktorého môže používateľ zvoliť jeden alebo viacero obrázkov, ktoré majú byť načítané.
                    file_path = QFileDialog.getOpenFileNames(self, 'Load inputs', self.__file_path,
                                                             'Image files (*.jpg *.png )')

                    # Premenná file_path predstavuje list obsahujúci na mieste prvého prvku list so zvolenými súbormi a
                    # na druhom prvku listu sa nachádza typ zvolených súborov. Ak je list zvolených súborov prázdny,
                    # program nevykoná žiadnu akciu v opačnom prípade je zavolaná metód triedy logickej vrstvy
                    # ktorá ako parameter preberá list obsahujúci cesty k súborom.
                    if len(file_path[0]) != 0:

                        # Ak používateľ zvolil nejaké súbory, budú poslané ako argument metódy. Výstup z metódy slúži
                        # ako indikátor, či bol vstup načítaný úspešne. Ak načítavanie prebehlo v poriadku, premenná
                        # error_message obsahuje hodnotu None, v opačnom prípade sa v premennej nachádza informácia
                        # o tom, aká chyba pri načítani nestala.
                        error_message = self.__logic_layer.handle_images_input(file_path[0])
                else:
                    return
            else:
                file_path = QFileDialog.getOpenFileName(self, 'Load input', self.__file_path,
                                                        'Text files (*.txt);;CSV files (*.csv)')[0]
                if file_path != '':
                    error_message = self.__logic_layer.load_points(file_path)
            # Ak sa pri načítaní nevyskytla žiadna chyba, program skryje prípadne zobrazené chybové hlášky. Ak sa
            # pri načítaní vstupov predsa len nejaká chyba objavila, informatívny text bude zobrazený pomocou
            # informačného label elementu.
            if error_message is not None:
                self.__load_ilabel.setText("<font color='red'>{}</font>".format(error_message))
                self.__load_ilabel.show()
            else:
                self.__load_ilabel.hide()
        else:
            self.__load_ilabel.setText("<font color='red'>You have to load model first!</font>")

    def save_model(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Uloženie modelu za podmienky, že bol nejaký načítaný.
        """
        # Otestuje sa, či bol už nejaký model neurónovej siete načítaný. Ak nie, je informačnom elemente vypísaná výzva
        # aby používateľ najskôr načítal model neurónovej siete, pred tým ako ho môže uložiť.
        if self.__keras_model is not None:
            file_path = ''

            # Ak bol nejaký model už načítaný, objaví sa dialógové okno, ktoré umožní používateľovi zvoliť miesto a 
            # názov pod ktorým chce model uložiť. Dialógové okno sa otvára v priečinku, z ktorého bol model
            # načítaný a ako preddefinovaný názov je nastavený názov načítaného modelu.
            QFileDialog.getSaveFileName(self, 'Save Keras model', ntpath.join(self.__file_path, self.__file_name),
                                        'Keras model (*.h5)')[0]

            # Ak používateľ nevyberie žiadne okno, program skryje prípadne staré hlásenia a metóda končí. Ak bola zadaná
            # cesta a názov súboru, sú tieto informácie uložené do premenných pre budúce otváranie a ukladanie.
            # následne je model uložený na zvolené miesto.
            if file_path != '':
                self.__file_path, self.__file_name = ntpath.split(file_path)
                self.__keras_model.save(file_path)
            self.__save_ilabel.hide()
        else:
            # V prípade, že žiaden model nebol načítaný, bude zobrazená výzva použivateľovy, aby tak učinil, pred tým
            # ako sa pokúsi model uložiť.
            self.__save_ilabel.show()
            self.__save_ilabel.setText('You have to load model first!')
            self.__save_ilabel.setStyleSheet("QLabel { color : red; }")


class MainGraphFrame(QWidget):
    """
    Popis
    --------------------------------------------------------------------------------------------------------------------
    Hlavné okno podstránky s grafmi. Okno vytvorí scrollovacie okno, do ktorého budú následne vkladané jednotlivé vrstvy
    s grafmi a ich ovládačmi. Vytvorí taktiež options okno, ktoré slúži na zobrazovanie informácii o zvolenem bode a
    možnosti nastavenia zobrazenia grafov v jednotlivých vrstvách.

    Atribúty
    --------------------------------------------------------------------------------------------------------------------
    :var self.__logic_layer:        obsahuje odkaz na vrstvu, ktorá sa zaoberá výpočtami a logickou reprezentáciou
                                    jednotlivých vrstiev a váh.
    :var self.__number_of_layers:   počet vrstiev načítanej neurónovej siete
    :var self.__neural_layers:      obsahuje referenciu na vrstvy neurónovej siete.
    :var self.__name_to_order_dict: každej vrstve je postupne prideľované poradové číslo vrstvy, tak ako ide od
                                    začiatku neurónovej siete. Ako kľúč je použitý názov vrstvy.
    :var self.__order_to_name_dict: podľa poradia je každej vrstve pridelené poradové číslo. Poradové číslo je kľúčom
                                    do dict. Jeho hodnotami sú názvy jedntlivých vrstiev.
                                    Ide o spätný dict k __name_to_order_dict.
    :var self.__active_layers:      referencia na list, ktorý obsahuje poradové čísla aktívnych vrstiev. Je zdieľaný
                                    s logickou vrstvou aplikácie.
    :var self.__options_frame:      okno, v ktorom sa zobrazujú možnosti pre zvolenú vrstvu.
    :var self.scrollable_frame:     Qt skrolovacie okno, obalom pre QtWidget scrollbar content.
    :var self.__scrollbar_content:  udržuje v sebe jednotlivé GraphFrame, ktoré obsahujú grafyy a ovládače váh.
    :var self.__graph_area_layout:  rozloženie komponentov v scrollbar_content widgete.
    :var self.__options_frame:      odkaz na okno s možnosťami zobrazenia vrstvy.
    :var self.__add_rm_layer_rc:    combobox na výber vrstvy. Po vybratí sa vrstva zobrazí a odstráni z comboboxu.
    """
    def __init__(self, *args, **kwargs):
        super(MainGraphFrame, self).__init__(*args, **kwargs)

        # Definovanie základných atribútov triedy.
        self.__logic_layer = None
        self.__number_of_layers = 0

        self.__neural_layers = None

        self.__name_to_order_dict = {}

        self.__order_to_name_dict = {}

        self.__active_layers = []

        # Definovanie grafických komponentov triedy.
        self.scrollable_frame = QScrollArea(self)

        self.__scrollbar_content = QWidget(self)

        self.__graph_area_layout = QHBoxLayout(self.__scrollbar_content)

        self.__options_frame = OptionsFrame(self)

        self.__add_rm_layer_rc = RemovingCombobox(self.__scrollbar_content)

        # Inicializácia grafických komponentov
        self.initialize_ui()

        # Skrytie panelu s možnosťami.
        self.__options_frame.hide_option_bar()

    def initialize_ui(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializácia, rozloženie a zobrazenie grafických komponentov.
        """
        # Základné rozmiestnenie v rámci wrappera hlavného okna.
        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.setContentsMargins(0, 0, 5, 0)
        horizontal_layout.setSpacing(0)

        # Nastavenie fixnej šírky pre panel s možnosťami. K fixnej šírke sme pristúpili z dizajnérskeho hľadiska. Pri
        # inej politike nastavovania šírky sa panel zväčšuje a zmenšuje podľa rozkliknutej karty, čo zanecháva
        # zlý grafický dojem. Daná šírka je príjemná na pohľad a väčšina grafických komponentov sa do nej
        # bez problémov zmestí.
        self.__options_frame.setFixedWidth(320)

        # Do hlavného rozmiestnenia sú pridané hlavné položky a to okno s možnosťami a hlavné okno pre zobrazovanie
        # grafov. Panel s možnosťami sa bude nachádzať na ľavej strane a scrollovacie okno bude zobrazené napravo.
        horizontal_layout.addWidget(self.__options_frame, alignment=QtCore.Qt.AlignLeft)
        horizontal_layout.addWidget(self.scrollable_frame)
        self.__options_frame.adjustSize()

        # Nastavenie parametrov pre skcrollovacie okno a nastavenie rozmiestnenie v rámci skrolovacieho okna.
        # Nové prvky v rámci scrollovacieho okna sa budú zobrazovať naľavo.
        self.scrollable_frame.setWidgetResizable(True)
        self.__graph_area_layout.setContentsMargins(0, 0, 0, 0)
        self.__graph_area_layout.setSpacing(0)
        self.__graph_area_layout.setAlignment(QtCore.Qt.AlignLeft)
        self.__scrollbar_content.setLayout(self.__graph_area_layout)

        # Nastavenie obaľovacieho elementu pre obsah scrollovacieho okna. Bez tohto elementu by nebolo možné v rámci
        # scrollovacieho okna scrollovať.
        self.scrollable_frame.setWidget(self.__scrollbar_content)

        # Nastavenie fixnej šírky AddRemoveCombobox elementu a jeho následné do rozmiestnenia v rámci obsahu scrollova-
        # cieho okna. Fixná šírka bola zvolená z dizajnérskeho hľadiska.
        self.__add_rm_layer_rc.setFixedWidth(412)
        self.__graph_area_layout.addWidget(self.__add_rm_layer_rc)

    def initialize(self, logic_layer: GraphLogicLayer):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inializuje atribúty triedy.
        Vytvorí predbežný zoznam názvov jednotlivých vrstiev. Po inicializácií AddRemoveComboboxFrame budú získane
        unikátne mená, ktoré sa použijú ako kľúče v slovníku.
        Na základe unikátnych mien je vytvorený slovník, ktorý prevádza meno vrstvy na jej poradové číslo a spätný
        slovník, ktorý prevádza poradové číslo na názov vrstvy.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param logic_layer: referencia na logickú vrstvu aplikácie.
        """
        # Inicializácia panelu s možnosťami. Predovšetkým ide o skrytie tohto elementu a nastavenie aktívnej vrstvy na
        # None, ak ide už o opätovnú inicializáciu.
        self.__options_frame.initialize()

        # Nastavenie referencie na logickú vrstvu a vyčistenie slovníkov.
        self.__logic_layer = logic_layer
        self.__order_to_name_dict = {}
        self.__name_to_order_dict = {}

        # Priradí referenciu na list aktívnych vrstiev z logickej vrstvy.
        self.__active_layers = self.__logic_layer.active_layers

        # Priradí referenciu na list odkazov na triedy typu NeuralLayer. Taktiež uloží do premennej počet vrstiev
        # neurónovej siete.
        self.__neural_layers = self.__logic_layer.neural_layers
        self.__number_of_layers = len(self.__neural_layers)

        # Vytvorenie predbežného zoznamu názvov vrstiev na základe mien v keras modeli. Tento zoznam môže obsahovať
        # duplicity.
        layer_name_list = []
        for i in range(self.__number_of_layers):
            layer_name_list.append(self.__neural_layers[i].layer_name)

        # Inicializácia AddRemoveComboboxFrame. Metóda tejto triedy navracia list unikátnych názvov vrstiev.
        # Unikátne názvy su použité ako kľúče v slovníku a sú priradené príslušným vrstvám.
        unique_name_list = self.__add_rm_layer_rc.initialize(layer_name_list, self.show_layer,
                                                                 'Add layer', True, 'Select layer')
        # Vytvorenie slovníku s unikátnymi názvami vrstiev. Súčasne sú vrstvám priradené nové mená v prípade ak sa medzi
        # pôvodnými mentami objavili duplicity. Tieto mená totiž zohrávajú úlohu identifikátorov na pridávanie a
        # odoberanie vrstiev.
        for i, layerName in enumerate(unique_name_list):
            self.__neural_layers[i].layer_name = layerName
            self.__order_to_name_dict[i] = layerName
            self.__name_to_order_dict[layerName] = i

        # Ak je počet aktívnych vrstiev menší ako je počet všetkých vrstiev, je zobrazený AddRemoveComobobx, ktorý
        # umožňuje pridávanie vrstiev na základe ich mena. V opačnom prípade je tento element skrytý.
        if len(self.__active_layers) < self.__number_of_layers:
            self.__add_rm_layer_rc.show()
        else:
            self.__add_rm_layer_rc.hide()

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
        # Z premmennej si vyberieme názov vrstvy, ktorá bola zvolená. Na základe názvu získame index prislúchajúcej
        # vrstvy.
        layer_name = layer_tuple[1]
        layer_number = self.__name_to_order_dict[layer_name]

        # Otestuje, či je číslo vrstvy valídne.
        if 0 <= layer_number < self.__number_of_layers:
            layer_to_show = None

            # Inicializácia vrstvy na zobrazenie. V rámci nej sa na základe indexu vyberie vrstva neurónovej siete.
            # Pomocou metódy, sa vytvorí grafická reprezentácia vrstvy a odkaz na ňu je vrátený vo forme výstupnej
            # hodnoty metódy. Tejto metóde je nastavená minimálna šírka, kvôli tomu, aby sa postupným pridávaním
            # ďalších vrstiev zobrazil scrollbar. Bez nastavenia minimálnej šírky by po pridaní ďalších vrstiev
            # boli tieto elementy neprirodzene splošťované. Vytvorená grafická reprezentácia je nakoniec vlo-
            # žená za všetky predchádzajúce vrstvy a pred grafické vyobrazenie AddRemoveComobox elementu.
            if layer_number < self.__number_of_layers:
                layer_to_show = self.__neural_layers[layer_number]
                graph_frame_widget = layer_to_show.create_graph_frame(self.show_layer_options_frame, self.hide_layer)
                graph_frame_widget.setMinimumWidth(412)
                self.__graph_area_layout.insertWidget(self.__graph_area_layout.count() - 1, graph_frame_widget)

            # Poradové číslo vrstvy je následne vložené do listu aktívnych vrstiev. Tento list je použíaný pre efektí-
            # vnejšie vykonávanie funkcií ako sú update alebo prekreslenie, pretože niektoré zmeny sa prejavujú len
            # na nasledujúcich vrstvách a preto by bolo zbytočné vykonávať určité operácie pre vrstvy, ktoré ostanú
            # bez zmeny. Po pridaní vrstvy do listu aktívnych vrstiev je z AddRemoveComobox inštancie odobratá
            # položka s menom práve zobrazenej vrstvy.
            self.__active_layers.append(layer_number)
            self.__add_rm_layer_rc.hide_item(layer_name)

            # Novo pridanej vrstve sú priradené hodnoty výstupu na danej vrstve, ktoré má zobrazovať. Po priradení bodov
            # sú na vrstve aplikované zmeny, je použitá konfigurácia
            self.__logic_layer.set_points_for_layer(layer_number)
            layer_to_show.apply_displayed_data_changes()
            layer_to_show.use_config()
            layer_to_show.redraw_graph_if_active()

            # Ak je počet aktívnych vrstiev rovný celkovému počtu vrstiev je skrytý panel pre pridávanie nových vrstiev.
            if len(self.__active_layers) == self.__number_of_layers:
                self.__add_rm_layer_rc.hide()

    def hide_layer(self, layer_number: int):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Skryje vrstvu, podľa jej poradového čísla.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layer_number: číslo vrstvy, ktorá má byť skrytá
        """
        # Najskôr sa skontroluje, či sa vrstva s daným poradovým číslom nachádza medzi aktívnymi vrstvami. Ak áno,
        # na základe poradového čísla sa nájde príslušné meno vrstvy a odkaz z listu odkazov na neurónové vrstvy.
        # Zobrazovaný graf pre danú vrstvu je spolu s ovládačom váh vyčistený a je mu priradená hodnota None,
        # aby zanikli referencie na tento element a garbage collector mohol uvolniť prostriedky používané
        # týmto elementom.
        if layer_number in self.__active_layers:
            layer_name = self.__order_to_name_dict[layer_number]
            layer = self.__neural_layers[layer_number]
            layer.graph_frame.clear()
            layer.graph_frame = None

            # Následne je číslo vrstvy odstránené z listu aktívnch vrstiev. A názov vrstvy je opätovne pridaný do
            # AddRemoveCombobox elementu, pre opätovné zobrazenie vrstvy. Ak je počet zobrazených vrstiev menší
            # ako počet všetkých vrstiev, je zobrazený AddRemoveCombobox element. Ak aktuálne skrytá vrstva
            # nastavená v rámci panelu možností, je tento panel skrytý.
            self.__active_layers.remove(layer_number)
            self.__add_rm_layer_rc.show_item(layer_name)
            if len(self.__active_layers) < self.__number_of_layers:
                self.__add_rm_layer_rc.show()
            if self.__options_frame.active_layer == layer:
                self.__options_frame.hide_option_bar()

    def show_layer_options_frame(self, layer_number):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zobrazí na panele, možnosti zobrazenia pre požadovanú vrstvu.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layer_number: číslo vrstvy, ktorej možnosti majú byť zobrazené.
        """
        # Inicializuje panel možnosti konfiguráciou vrstvy, ktorej čislo bolo poslané ako argument.
        if 0 <= layer_number < self.__number_of_layers:
            layer = self.__neural_layers[layer_number]
            self.__options_frame.initialize_with_layer_config(layer, layer.config)

    def apply_changes_on_options_frame(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Aplikuje zmeny na panel možností.
        """
        self.__options_frame.update_selected_config()

    def update_active_options_layer(self, start_layer):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Aplikuje zmeny na panel možností, ak je splnená podmienka

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param start_layer: číslo vrstvy, na ktorej sa uskutočnila zmena.
        """
        self.__options_frame.update_active_options_layer(start_layer)

    def update_layer_if_active(self, neural_layer_ref):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Aplikuje zmeny na panel možností, ak je argument rovný práve zobrazovanej vrstve na panely možností.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param neural_layer_ref: referencia na vrstvu.
        """
        if neural_layer_ref == self.__options_frame.active_layer:
            self.apply_changes_on_options_frame()

    @property
    def logic_layer(self):
        return self.__logic_layer

    @logic_layer.setter
    def logic_layer(self, new_ref):
        self.__logic_layer = new_ref


class OptionsFrame(QWidget):
    def __init__(self, *args, **kwargs):
        """"
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Obsahuje možnosti zobrazenia pre jednotlivé vrstvy. V rámci nej je možne navoliť zobrazované súradnice pre
        jednotlivé metódy, ofarbiť body na základe ich label, povoliť zobrazenie mriežky, ak je to možné,
        uzmaknutie pohľadu v grafe. Je možné aj zvoliť redukciu priestoru a zobraziť požadovaný
        PCA komponent alebo použiť metódu t-SNE.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__labels_entries_list:  list vstupnov pre zadávanie názvov osi v grafe aktívnej vrstvy.
        :var self.__cords_entries_list:   list vstupov na zadávanie zobrazovaných súradníc.
        :var self.__tSNE_parameters_dict: obsahuje grafické prvky pre zadávanie parametrov pre výpočet metódy t-SNE.
        :var self.__changed_config:       referencia na konfiguračný slovník vrstvy, pre ktorú sa zobrazujú možnosti.
        :var self.__active_layer:         odkaz na aktívnu vrstvu, pre ktorú sú aktuálne zobrazené možnosti.
        :var self.__layer_name_label:     zobrazuje meno aktívnej vrstvy, pre ktorú sú zobrazované možnosti.
        :var self.__bar_wrapper:          obaľovací element, obsahuje jednotlivé podskupiny možností.
        :var self.__choose_cords_gbox:    obaľovací element skupiny možností pre nastavenie zobrazovaných súradníc.
        :var self.__possible_cords_label: zobrazuje informáciu o rozsahu súradníc, ktoré môžu byť zobrazené.
        :var self.__choose_label_gbox:    obaľovací element pre skupinu možností zodpovednú za názvy osí grafu.
        :var self.__graph_view_gbox:      obaľovací element pre skupinu možností zodpovednú za možnosti vzhľadu a
                                          zobrazovaných prvkov vo vykresľovanom grafe.
        :var self.__color_labels_cb:      checkbox, ktorý zachytáva, či je požoadované aby v zobrazovanom grafe boli
                                          body ofarebené na základe príslušnosti k triede.
        :var self.__lock_view_cb:         checkbox, ktorý značí, či májú byť pri prekreslení grafu zanechaný posledne
                                          nastavený zoom, offset a prípadne rotácia.
        :var self.__3d_view_cb:           checbox, ktorý zachytáva, či majú byť vykreslené body v 2D alebo 3D grafe.
        :var self.__show_polygon:         checbox, ktorý značí, či majú byť vykreslené hrany polygónu priestoru, ktorý
                                          lemuje zadané body, ak je vstup neurónovej vrstvy 2D alebo 3D.
        :var self.__dim_reduction_gbox:   obaľovací element, zahŕňajúci možnosti pre redukciu priestoru.
        :var self.__actual_reduc_method:  label element, ktorý oboznamuje používateľa s aktuálne 
                                          použitou metódou redkucie priestoru.
        :var self.__no_method_rb:         radiobutton, ktorý sĺuži na nastavenie požadovanej metódy na redukciu
                                          priestoru, konkrétne použitie metódy: No method
        :var self.__PCA_rb:               radiobutton, ktorý sĺuži na nastavenie požadovanej metódy na redukciu
                                          priestoru, konkrétne použitie metódy: PCA
        :var self.__t_SNE_rb:             radiobutton, ktorý sĺuži na nastavenie požadovanej metódy na redukciu
                                          priestoru, konkrétne použitie metódy: t-SNE
        :var self.__pca_info_gbox:        obaľovací element, ktorý obaĺuje prvky zobrazujúce informácie o vykonanej 
                                          PCA analýze. 
        :var self.__var_expl_lb:          listbox, ktorý zobrazuje informácie o vysvetlenej variabilite jednotlivými
                                          komponentami na zvolenej vrstve, ak bola použitá metóda PCA
        :var self.__load_scores_lb:       listbox, ktorý zobrazuje informácie o vpyve jednotlivých vstupov na zvolený
                                          hlavný komponent, ak bola použitá metóda PCA.
        :var self.__t_sne_parameter_gbox: obaľovací element, ktorý v sebe zahŕňa grafické prvky pre zadávanie parametrov
                                          použitých aplikovanie metódy t-SNE.
        :var self.__use_method_btn:       tlačidlo, na použitie zvolenej metódy pomocou radioboxu.
        :var self.__cur_used_method:      názov práve používanej metódy redukcie priestor.
        """
        super(OptionsFrame, self).__init__(*args, **kwargs)
        # Nastavenie veľkosti fontu pre celý panel.
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setFont(font)

        # Definovanie základných atribútov.
        self.__labels_entries_list = []
        self.__cords_entries_list = []
        self.__tSNE_parameters_dict = {}
        self.__changed_config = None
        self.__active_layer = None

        # Definovanie grafických elementov.

        # Definícia obaľovacieho elementu pre všetky podskupiny možností a definovanie label elementu, ktorý bude
        # zobrazovať meno vrstvy, ktorej nastavenia môžeme meniť.
        self.__bar_wrapper = QGroupBox('Layer options')
        self.__layer_name_label = QLabel('Layer name')

        # Obaľovací element podskupiny zodpovednej za výber zobrazovaných súradníc. Taktiež je definovaný aj label
        # element, ktorý bude zobrazovať rozsah možných súradníc, ktoré sa dajú zobraziť.
        self.__choose_cords_gbox = QGroupBox('Displayed cords')
        self.__possible_cords_label = QLabel('Possible cords')

        # Obaľovací element pre podskupinu grafický prvkov starajúcich sa o zmeny názvu osí grafu.
        self.__choose_label_gbox = QGroupBox('Axis name')

        # Definíci elementu, ktorý obaľuje možnosti týkajúce sa vzhľadu grafu, zobrazovaného obsahu a pohľadu na graf.
        # Sú zadefinované checkboxy pre jednotlivé možnosti.
        self.__graph_view_gbox = QGroupBox('Graph view options')
        self.__color_labels_cb = QCheckBox('Color according to label')
        self.__lock_view_cb = QCheckBox('Lock view')
        self.__3d_view_cb = QCheckBox('3D view')
        self.__show_polygon_cb = QCheckBox('Show space grid')

        # Definícia obalu držiaceho grafické prvky, slúžiace na zmenu metódy redukcie priestoru. Taktiež sú zadefinované
        # aj grafické prvky, ktoré zobrazujú informácie o určitých parametroch métod, prípadne panel pre navolenie
        # parametrov potrebných pre použitie metódy.
        self.__dim_reduction_gbox = QGroupBox('Dimension reduction')
        self.__actual_reduc_method = QLabel('Actual used: No method')
        self.__no_method_rb = QRadioButton('No method')
        self.__PCA_rb = QRadioButton('PCA')
        self.__t_SNE_rb = QRadioButton('t-SNE')
        self.__pca_info_gbox = QGroupBox('PCA information')
        self.__var_expl_lb = QListWidget()
        self.__load_scores_lb = QListWidget()
        self.__t_sne_parameter_gbox = QGroupBox('t-SNE parameters')
        self.__use_method_btn = QPushButton()
        self.__cur_used_method = 'No method'

        # Grafická inicializácia panelu možností.
        self.initialize_ui()

    def initialize_ui(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializácia jednotlivých grafických prvkov.
        """
        # Vytvorenie inštancie rozmiestnenia, ktoré bude použité na rozmiestnenie jednotlivých podskupín elementov.
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(5, 0, 5, 5)
        main_layout.addWidget(self.__bar_wrapper)

        # Toto rozmiestnenie sa stará o podskupiny možností nastavení grafu. Rozmiestnenie je následne nastavené obalo-
        # vaciemu elementu.
        options_groups_layout = QVBoxLayout()
        options_groups_layout.setAlignment(QtCore.Qt.AlignTop)
        options_groups_layout.setSpacing(0)
        options_groups_layout.setContentsMargins(0, 0, 0, 0)
        self.__bar_wrapper.setLayout(options_groups_layout)

        # Kvôli vizuálu je elementu, ktorý zobrazuje názov vrstvy pridaný margin.
        self.__layer_name_label.setMargin(10)

        # Vytvorenie fontu a následné priradenie fontu elementu, ktorý zobrazuje meno vrstvy. Font je nastavený na urči-
        # tú veľkosť a je aj nastavená možnosť hrubšieho písma. Po nastavení fontu je element vložený do rozmiestnenia.
        layer_name_font = QtGui.QFont()
        layer_name_font.setPointSize(10)
        layer_name_font.setBold(True)
        self.__layer_name_label.setFont(layer_name_font)
        options_groups_layout.addWidget(self.__layer_name_label, alignment=QtCore.Qt.AlignCenter)

        # Vytvorenie rozmiestnenia pre kontajner držiaci grafické elementy pre zmenu zobrazovaných súradníc a jeho ná-
        # sledné nastavenie tomuto kontajneru. Následne je do tohto rozmiestnenia vložený element na zobrazovanie
        # rozsahu súradníc, ktoré je možné zobraziť. Taktiež je tento kontajner vložený do rozmiestnenia
        # pre podskupiny nastavení.
        choose_cords_layout = QVBoxLayout()
        choose_cords_layout.setAlignment(QtCore.Qt.AlignTop)
        self.__choose_cords_gbox.setLayout(choose_cords_layout)
        choose_cords_layout.addWidget(self.__possible_cords_label, alignment=QtCore.Qt.AlignHCenter)
        options_groups_layout.addWidget(self.__choose_cords_gbox)

        self.__possible_cords_label.setContentsMargins(0, 0, 0, 0)

        # Do rozmiestnenia pre podskupiny je vložený obaľovací element pre podskupinu prvkov zodpovedných za zmenu ozna-
        # čení jednotlivých osí. Následje je vytvorená inštancia rozmiestnenia pre tento obaľovací element a v rámci
        # rozmiestnenia je nastavené zarovnávanie naľavo.
        options_groups_layout.addWidget(self.__choose_label_gbox)
        choose_labels_layout = QVBoxLayout()
        self.__choose_label_gbox.setLayout(choose_labels_layout)
        choose_labels_layout.setAlignment(QtCore.Qt.AlignTop)

        # V nasledujúcom cykle sú vytvorené a inicializované grafické prvky pre vstup údajov od používateľa. Ide o vstu-
        # pné prvky pre zadávanie požadovaných zobrazovaných súradníc a vstupy na zadávanie označení jednotlivých osí.
        # Ide o grafické prvky typu RewritableLabel, ktorým je nastavený identifikátor, názov, preddefinovaná hodnota
        # a funkcia, ktorá sa zavolá pri aktivácií prvku. Prvky sú postupne pridávané do rozmiestení, ku ktorým
        # patria.
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

        # Do rozmiestnenia podskupín možností je pridaná podskupina zaoberajúca sa vzhľadom a zobrazovaným obsahom gra-
        # fu. Je vytvorené ďalšie rozmiestnenie, ktoré je nastavené obaľovaciemu elementu tejto podskupiny.
        # V rámci tohto rozmiestnenia budú grafické elementy zarovnávané naľavo.
        options_groups_layout.addWidget(self.__graph_view_gbox)
        view_options_layout = QVBoxLayout()
        self.__graph_view_gbox.setLayout(view_options_layout)
        view_options_layout.setAlignment(QtCore.Qt.AlignTop)

        # Checboxom pre jednotlivé možnosti je nastavená funkcia, ktorá sa zavolá po kliknutí na možnosť. Po nastavení
        # sú jednotlivé checkboxy pridané do rozmiestnenia.
        self.__color_labels_cb.toggled.connect(self.on_color_label_check)
        self.__lock_view_cb.toggled.connect(self.on_lock_view_check)
        self.__3d_view_cb.toggled.connect(self.on_3d_graph_check)
        self.__show_polygon_cb.toggled.connect(self.on_show_polygon_check)
        view_options_layout.addWidget(self.__color_labels_cb)
        view_options_layout.addWidget(self.__lock_view_cb)
        view_options_layout.addWidget(self.__3d_view_cb)
        view_options_layout.addWidget(self.__show_polygon_cb)

        # Do rozmiestnenia pre podskupiny možností je pridaná podskupina obsahujúca možnosti na redukciu priestoru.
        # Je vytvorená ďalšia inštancia vertikálneho rozmiestnenia, ktoré je nastavené tomuto elementu.
        options_groups_layout.addWidget(self.__dim_reduction_gbox)
        dim_reduction_layout = QVBoxLayout()
        dim_reduction_layout.setContentsMargins(0, 0, 0, 0)
        dim_reduction_layout.setAlignment(QtCore.Qt.AlignTop)
        self.__dim_reduction_gbox.setLayout(dim_reduction_layout)

        # Elementu, ktorý informuje o práve používanej metóde redukcie priestoru je nastavený margin a je element je
        # pridaný do rozmiestnenia obaľovacieho elementu. Súčasne mu je nastavené aj zarovnanie.
        self.__actual_reduc_method.setMargin(10)
        dim_reduction_layout.addWidget(self.__actual_reduc_method,
                                       alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        # Vytvorenie obaľovacieho elementu, ktorý zaručí aby pri zakliknutí nejakého radio buttonu boli ostatne odzna-
        # čené. Taktiež je vytvorená rozmiestnenie, do ktorého budú vkladané jednotlivé radio buttony na voľbu metódy
        # použitej na redukciu priestoru. Do rozmiestnenia podskupiny na redukciu priestoru je pridané rozmiestnenie
        # pre radio buttony. Následne je radio buttononom nastavená funkcia, ktorá sa má spustiť po kliknutí. Na
        # záver je nastavené zakliknutie na radio buttone pre možnosť No method.
        radio_button_group = QButtonGroup()
        radio_button_layout = QHBoxLayout()
        radio_button_layout.setContentsMargins(5, 0, 5, 5)
        radio_button_layout.setAlignment(QtCore.Qt.AlignLeft)
        dim_reduction_layout.addLayout(radio_button_layout)
        self.__no_method_rb.toggled.connect(self.on_method_change)
        self.__PCA_rb.toggled.connect(self.on_method_change)
        self.__t_SNE_rb.toggled.connect(self.on_method_change)
        self.__no_method_rb.setChecked(True)

        # Radio buttony sú popridávané do skupiny tlačidiel a taktieź aj do rozmiestnenia v rámci obaľovacieho elementu
        # týchto tlačidiel.
        radio_button_group.addButton(self.__no_method_rb)
        radio_button_layout.addWidget(self.__no_method_rb)
        radio_button_group.addButton(self.__PCA_rb)
        radio_button_layout.addWidget(self.__PCA_rb)
        radio_button_group.addButton(self.__t_SNE_rb)
        radio_button_layout.addWidget(self.__t_SNE_rb)

        # Listboxu zobrazujúcemu variabilitu vysvetlenú jednotlivými komponentami je priradená funkcia, ktorá sa zavolá
        # po kliknutí na komponent. Obaľovaciemu elementu listboxov je nastavená maximálna výška. Listbox je pridaný
        # do rozmiestnenia obaľovacieho elementu pre tento listbox so zarovnaním nahor, toto zarovnanie spôsobí, že
        # sa natiahne po maximálnu výšku obaľovacieho elementu.
        self.__var_expl_lb.currentRowChanged.connect(self.show_PCA_loading_scores)
        self.__pca_info_gbox.setMaximumHeight(200)
        dim_reduction_layout.addWidget(self.__pca_info_gbox, alignment=QtCore.Qt.AlignTop)

        # Vytvorenie rozmiestnenia pre podskupinu obsahujúcu informácie o výstupe metódy PCA na danej vrstve.
        pca_info_layout = QHBoxLayout(self.__pca_info_gbox)
        pca_info_layout.setAlignment(QtCore.Qt.AlignTop)
        pca_info_layout.setContentsMargins(0, 10, 0, 0)

        # Vytvorenie obaľovacieho elementu pre listbox vysvetľujúci variabilitu a pridanie tohto kontajnera do
        # rozmiestnenia informačnej podskupiny.
        var_expl_gbox = QGroupBox('PC variance explanation')
        pca_info_layout.addWidget(var_expl_gbox)

        # Vytvorenie rozmiestnenia pre obaľovací element listboxu zobrazujúceho vysvetlenú variabilitu.
        layout_var_expl = QVBoxLayout(var_expl_gbox)
        layout_var_expl.setAlignment(QtCore.Qt.AlignTop)
        layout_var_expl.setContentsMargins(0, 0, 0, 0)

        # Listboxu vysvetľujúcej variability pre jednotlivé komponenty je nastavená maximálna šírka a listbox je
        # pridaný do rozmiestnenia.
        self.__var_expl_lb.setMaximumWidth(150)
        layout_var_expl.addWidget(self.__var_expl_lb)

        # Vytvorenie obaľovacieho elementu pre listbox, pojednávajúci o vplyve vstupov na zvolený hlavný komponent.
        loading_score_gbox = QGroupBox('Loading scores')
        pca_info_layout.addWidget(loading_score_gbox)

        # Vytvorenie rozmiestnenia pre obaľovací element, listboxu s hodnotami vpyvu vstupov na jednotlivé komponenty.
        # Následne je listbox pridaný do rozmiestnenia.
        loading_score_layout = QVBoxLayout(loading_score_gbox)
        loading_score_layout.setAlignment(QtCore.Qt.AlignTop)
        loading_score_layout.setContentsMargins(0, 0, 0, 0)
        self.__load_scores_lb.setMaximumWidth(150)
        loading_score_layout.addWidget(self.__load_scores_lb)

        # Pridanie obaľovacieho elementu pre vsupné prvky na definíciu parametrov metódy t-SNE do rozmiestnenia podsku-
        # piny možností pre redukciu priestoru.
        dim_reduction_layout.addWidget(self.__t_sne_parameter_gbox, alignment=QtCore.Qt.AlignTop)
        t_sne_par_layout = QVBoxLayout(self.__t_sne_parameter_gbox)
        t_sne_par_layout.setAlignment(QtCore.Qt.AlignTop)

        # Vytvorenie listov obsahujúcich identifikátor a názov pre inštancie triedy RewritableLabel. Identifikátor je
        # zhodný s kľúčom do konfiguračnej premennej pre výpočet metódy t-SNE.
        t_sne_parameter_id_list = ['n_components', 'perplexity', 'early_exaggeration', 'learning_rate', 'n_iter']
        t_sne_parameter_label = ['Number of components:', 'Perplexity:', 'Early exaggeration:', 'Learning rate:',
                                 'Number of iteration:']
        for i in range(len(t_sne_parameter_id_list)):
            t_sne_parameter_rw = RewritableLabel(t_sne_parameter_id_list[i], t_sne_parameter_label[i], '-',
                                                 self.validate_t_sne_entry)
            t_sne_par_layout.addWidget(t_sne_parameter_rw, alignment=QtCore.Qt.AlignLeft)
            self.__tSNE_parameters_dict[t_sne_parameter_id_list[i]] = t_sne_parameter_rw

        # Nastavenie textu tlačidla, jeho pridanie do rozmiestnenia a nastavenie funkcie, ktorá bude zavolaná po kliknu-
        # tí na tlačidlo.
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

    def initialize_with_layer_config(self, neural_layer: NeuralLayer, config: dict):
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
        Nastavenie názvov jednotlivých vstupov pre časť s označením osi grafov na základe načítaného configu.
        """
        # Zo zvoleného konfigu je vybratý parameter pojednávajúci o maximálnej možnej zobrazenej dimenzií. Tá môže byť
        # od 1 až po 3. Na základe tohto údaja bude zobrazený zodpovedajúci počet vstupov pre označenie osí grafov.
        # Z listu grafov sú postupne vyberané odkazy na grafické vstupy a je nastavená ich hodnota na základe
        # základných alebo posledne nastavených názvov osí. Element je následne zobrazený.
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
        # Na základe pravdivostnej hodnoty sú začiarknuté alebo odznačené jednotlivé môžnosti pre vzhľad grafu.
        self.__color_labels_cb.setChecked(self.__changed_config['color_labels'])
        self.__lock_view_cb.setChecked(self.__changed_config['locked_view'])

        # Z konfiguračnej premennej je zistený maximálny počet viditeľných dimenzií. Na základe tohto údaju su niektoré
        # checkboxy zobrazené alebo ponechané naďalej skryté.
        number_of_possible_dim = self.__changed_config['max_visible_dim']
        if number_of_possible_dim >= 3:
            self.__3d_view_cb.setChecked(self.__changed_config['draw_3d'])
            self.__3d_view_cb.show()

        # Podľa toho či je vstup do siete 2D alebo 3D je možné zobraziť mriežku ohraničujúcu body. Ak nie táto možnosť
        # nie je príspustná, checkbox ostane skrytý. Ak je táto možnosť prípustná, je nastavená posledne zvolená
        # možnosť, ktorú používateľ zadal.
        if self.__changed_config['possible_polygon']:
            self.__show_polygon_cb.setChecked(self.__changed_config['show_polygon'])
            self.__show_polygon_cb.show()

        # Ak boli pre body načítané labels, naskytne sa možnosť ofarbiť body na základe príslušnosti k triede. Ak žiadne
        # labels neboli načítané, zostane táto možnosť vypnutá a naďalej skrytá.
        if self.__active_layer.possible_color_labels:
            self.__color_labels_cb.setChecked(self.__changed_config['color_labels'])
            self.__color_labels_cb.show()

    def initialize_dimension_reduction_options(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie hodnôt jednotlivých vstupov pre časť s možnosťami pre redukciu priestoru.
        """
        # Je nastavený názov aktuálne použitej metódy. A do premennej je vložený názov posledne zvolenej metódy. Metóda
        # nemusí byť ani zvolená. Budú nastavené predbežné nastavenia pre túto metódu.
        self.set_actual_method_lable(self.__cur_used_method)
        config_selected_method = self.__changed_config['config_selected_method']

        # Všetky radio buttony zodpovedajúce metódam sú zbavené zakliknutia.
        self.__no_method_rb.setChecked(False)
        self.__PCA_rb.setChecked(False)
        self.__t_SNE_rb.setChecked(False)

        # Podľa posledne zakliknutej metódy je zvolený príslušný radio button.
        if config_selected_method == 'No method':
            self.__no_method_rb.setChecked(True)
        elif config_selected_method == 'PCA':
            self.__PCA_rb.setChecked(True)
        elif config_selected_method == 't-SNE':
            self.__t_SNE_rb.setChecked(True)

        # Ak je aktuálne použitá metóda PCA, sú vypísané aj informácie o výstupe metódy.
        if self.__cur_used_method == 'PCA':
            self.update_PCA_information()

        # Následne sú prednastavené hodnoty parametrov pre t-SNE pre danú vrstvu.
        self.initialize_t_sne_parameters()
        self.on_method_change()

    def initialize_t_sne_parameters(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Predvyplnenie parametrov pre metódu t-SNE.
        """
        # Z konfiguračnej premennej pre aktívnu vrstvu je vybraná podzložka pre s-SNE konfiguráciu. Z tejto konfigurácie
        # sú získané hodnoty pre aktuálne použité parametre a pre posledne nastavené parametre.
        t_sne_config = self.__changed_config['t_SNE_config']
        actual_used_config = t_sne_config['used_config']
        options_config = t_sne_config['options_config']

        # Z konfigurácie pre hraničné hodnoty parametrov je vybraný maximálny počet možných komponentov. Na základe
        # tohoto čísla je nastavený názov vstupného grafického prvku, aby bol používateľ informovaný o tom, koľko
        # komponentov môže maximálne použiť.
        number_of_components = t_sne_config['parameter_borders']['n_components'][2]
        self.__tSNE_parameters_dict['n_components'].set_label_name(
            f'Number of components (max {number_of_components}):')

        # Následne sú v cykle na základe kľúčov prednastavené hodnoty vstupov.
        for key in self.__tSNE_parameters_dict:
            rewritable_label = self.__tSNE_parameters_dict[key]
            rewritable_label.show_variable_label()
            rewritable_label.set_variable_label(options_config[key])

            # Ak sa aktuálne používaná hodnota parametra nerovná naposledy nastavenej hodnote parametra je tento
            # parameter označený ako zmenený no ešte nepoužitý. Vyznačuje sa to tým, že je názov vyznačený
            # červenou farbou a je pridaná hviezdička.
            if actual_used_config[key] == options_config[key]:
                rewritable_label.set_mark_changed(False)
            else:
                rewritable_label.set_mark_changed(True)

    def use_selected_method(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Funkcia, ktorá sa zavolá po zvolení metódy a klikutí na tlačidlo Use method. Na základe zvolenej metódy, nastaví
        v konfiguračnej premennej hodnotu aktuálnej metódy a taktiež prispôsobí vzhľad a hodnoty na panele možností.
        """
        if self.__active_layer is not None:
            # Je získaná metóda, podľa zakliknutého radio buttonu. Je definovaná premenná need_recalculation, aby
            # nedošlo k zbytočnému prepočítavaniu na vrstve, nedošlo k žiadne zmene ale napríklad bolo tlačidlo
            # viac krát stlačené.
            method = self.get_checked_method()
            need_recalculation = False

            # Ak je začiarknutá metóda t-SNE, zistí sa či boli zmenené hodnoty niektorých parametrov oproti naposledy
            # použitým. Ak áno, do premennej need_recalculation sa priradí hodnota True ak nie, priradí sa False.
            if method == 't-SNE':
                need_recalculation = self.apply_t_SNE_options_if_changed()

            # Ak sa zakliknutá metóda líši od aktuálne používanej metódy.
            if method != self.__changed_config['used_method']:
                # Ak je zvolená metóda PCA, je potrbné nastaviť zobrazované súradnice aby nedošlo ku chybe. Zoznam súra-
                # dníc získame na základe minimálnej hodnoty spomedzi výstupnej dimenzie, pretože nie je možné zobrazo-
                # vať viac hlavných komponentov ako je počet výstupných hodnôt, počtu vzoriek, pretože PCA vytvorí len
                # minimum(počtu výstupných príznakov, počet vzoriek) hlavných komponentov a čísla 3, pretože je možné
                # zobraziť maximálne 3 dimenzie. Na základe tohto čísa sa vytvorí list súradníc, ktorý je priadený
                # do konfiguračnej premennej. Ďalej sa nastaví premenná need_recalculation na True, pretože zmena
                # metódy zaručuje potrebu vykonania prepočtu a nastavenia súradníc. Na záver je priradená do
                # konfiguračného súboru priradená zvolená metóda a aj v rámci panelu možností je nastavená
                # hodnota pre premennú self.__cur_used_method značiaca práve používanú metódu.
                if method == 'PCA':
                    self.__changed_config['PCA_config']['displayed_cords'] = list(range(min(
                                                                            self.__changed_config['output_dimension'],
                                                                            self.__changed_config['number_of_samples'],
                                                                            3)))

                need_recalculation = True
                self.__changed_config['used_method'] = self.__cur_used_method = method
            # V prípade ak došlo k zmenám je potrebné aplikovať zmeny a prekresliť graf. Ak nedošlo k zmene, metóda kon-
            # čí. Podľa použitej metódy sú zobrazené a aktualizované aj dodatočné údaje.
            if need_recalculation:
                self.__changed_config['apply_changes'] = True
                self.__active_layer.use_config()
                self.__active_layer.redraw_graph_if_active()
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

    def update_selected_config(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Aktualizovanie jednotlivých možností na základe configu aktívnej vrstvy.
        """
        # Podľa aktívneho konfigu sú nastavené hodnoty grafických prvkov. Je nastavené meno vrstvy, sú skryté všetky
        # časti panelu, ktorých zobrazenie závisí od určitých podmienok. Z konfiguračného súboru je priradené metó-
        # da, používana na redukciu priestoru.
        if self.__active_layer is not None and self.__changed_config is not None:
            self.__cur_used_method = self.__changed_config['used_method']
            self.hide_option_bar()
            self.__bar_wrapper.show()
            self.__layer_name_label.setText(str(self.__changed_config['layer_name']))

            # Inicializácia grafických komponentov jednotlivých podskupín v závislosti od aktívnej konfiguračnej premen-
            # nej.
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
        # Listboxy držiace informačné hodnoty sú vyčistené a z konfiguračnej premennej je vybraný séria, ktorá obsahuje
        # informácie o tom, akú variabilitu vysvetľujú jednotlivé hlavné komponenty.
        self.__var_expl_lb.clear()
        self.__load_scores_lb.clear()
        variance_series = self.__changed_config['PCA_config']['percentage_variance']
        if variance_series is not None:
            # Do premennej pc_labels je priradený index série, zodpovedajúci názvu kompnentu v tvare PC{poradové číslo}.
            # Tieto hodnoty sú postupne vkladané do listboxu, spolu s príslušnou percentuálnou hodnotou variability,
            # ktorú daný komponent vyjadruje.
            pc_labels = variance_series.index
            for i, label in enumerate(pc_labels):
                self.__var_expl_lb.insertItem(i, '{}: {:.2f}%'.format(label, round(variance_series[label], 2)))

            # V informačnom listboxe, ktorý nesie informáciu o vplyve jednotlivých výstupov na komponenty sú zobrazené
            # hodnoty pre prvý hlavný komponent.
            self.show_PCA_loading_scores(0)

    def show_PCA_loading_scores(self, component_number):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Na základe parametra component_number vloží do listboxu informácie o vplyve výstupov na hlavný komponent,
        ktorého číslo bolo zadané ako argument.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param component_number: poradové číslo komponentu, o ktorom majú byť zobrazené informácie
        """
        # Testuje sa, či bolo zadané valídne číslo komponentu. Následne je listbox vyčistený, ak náhodou obsahoval staré
        # hodnoty. Následne je z konfiguračnej premennej na zákalde čísla komponentu o ktorom požadujeme informácie z
        # premennej typu dataframe získaný riadok, obsahujúci vplyv výstupov na požadovaný hlavný komponent.
        if component_number > -1:
            self.__load_scores_lb.clear()
            loading_scores = self.__changed_config['PCA_config']['largest_influence'][component_number]

            # Hodnoty sú zoradené podľa vpyvu zostupne, čiže na vrchu je uvedený výstup, ktorý ma daný komponent najvä-
            # čší vplyv. Zo zoradeného dataframe sa vyberú zoradené indexy. Tie sú v cykle vypísané spoločne s mierou
            # ich vplyvu na hlavný komponent.
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
        # Pre istotu sa skontroluje, či je aktívna nejaká vrstva a tak isto, či je prítomná konfiguračná premenná pre
        # danú vrstvu. Následne sa zistí použitá metóda. Ak ide o metódu PCA, mohlo dôjsť k zmenám vo vyjadrenej var-
        # iabilite, prípadne vplyvu výstupu na hlavný komponent. K tomu mohlo dôjsť ak došlo k zmene na v poradí
        # nižšej alebo rovnakej vrstve.
        if self.__active_layer is not None and self.__changed_config is not None:
            actual_method = self.__changed_config['used_method']
            if actual_method == 'PCA' and start_layer <= self.__active_layer.layer_number:
                self.update_PCA_information()

    def apply_t_SNE_options_if_changed(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda priradí do konfiguračnej premennej hodnoty nastavených parametrov a testuje, či došlo k zmene oproti
        posledne nastaveným hodnotám. Ak došlo k zmene aspoň pri jednom parametre je premennej changed nastavená
        hodnota True. Výsledná hodnota tejto premennej je návratovou hodnotou metódy.

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return: vracia hodnotu
        """
        changed = False
        if self.__changed_config is not None:
            # Z konfiguračnej premennej získame konfiguráciu pre metódu t-SNE. Do premenných si uložíme aktuálne použí-
            # vanú a posledne použitú konfiguráciu. Následne tento prechádzame tieto konfigurácie v cykle a zisťujeme
            # či sa líšia. Ak áno, nastaví sa nová hodnota parametra ako používaná hodnota parametra a premenná
            # changed zmení svoju hodnotu na True, čo symbolizuje, že došlo k zmene.
            t_sne_config = self.__changed_config['t_SNE_config']
            used_config = t_sne_config['used_config']
            options_config = t_sne_config['options_config']
            for key in used_config:
                if used_config[key] != options_config[key]:
                    changed = True
                    used_config[key] = options_config[key]
            # Ak došlo k zmene, je potrebné na základe parametra pre metódu t-SNE nastaviť súradnice, ktoré sa majú
            # zobraziť na jednotlivých osiach.
            if changed:
                t_sne_config['displayed_cords'] = list(range(used_config['n_components']))
                self.set_entries_not_marked(self.__tSNE_parameters_dict.values())
        return changed

    def set_cords_entries_according_chosen_method(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie textu a používaných hodnôt na základe zvolenej metódy.
        :return:
        """
        # Na základe aktuálne použitej metódy an redukciu priestoru sú definované hodnoty, ktoré sa majú nastaviť a
        # zobraziť na paneli možností v jednotlivých podskupinách.
        if self.__cur_used_method == 'No method':

            # Ak je aktuálne zvolená metóda No method pokračuje sa touto vetvou. Následne sa nastavia hodnoty pre
            # premenné, podľa toho, či sú daná vrstva obsahuje feature mapy alebo nie.
            if self.__changed_config['has_feature_maps']:

                # Ak aktívna vrstva obsahuje feature mapy, sú názvy a hodnoty vstupov preddefinované inak, pretože
                # mapy sú dvojrozmerné a pre lepšiu predstavu a pohdlnejšie zobrazovanie je vhodné použiť systém
                # zobrazovania (riadok, stĺpec). Informácia o prípustných súradniciach je vložená do premennej
                # cords_label text. Umožňuje zobraziť súradnice od 0 po prvé číslo mínus 1 pre voľbu riadku
                # a srúadnicu od 0 po druhé číslo mínus 1 pre voľbu stĺpca.
                output_shape = self.__changed_config['output_shape']
                cords_label_text = 'Possible cords: {}x{}'.format(output_shape[-3], output_shape[-2])
                tmp_displayed_cords_tuples = list(zip(self.__changed_config['no_method_config']['displayed_cords'][0],
                                                      self.__changed_config['no_method_config']['displayed_cords'][1]))
                # Na základe naposledy nastanených zobrazovaných súradníc je naplnená premenná displayed cords.
                displayed_cords = []
                for x, y in tmp_displayed_cords_tuples:
                    displayed_cords.append(f'{x}, {y}')
            else:
                # Ak vrstva nebsahuje feature mapy, je informatívny text nastavený na prípustné zobrazené hodnoty.
                cords_label_text = 'Possible cords: 0-{}'.format(self.__changed_config['output_dimension'] - 1)
                displayed_cords = self.__changed_config['no_method_config']['displayed_cords']
            entry_names = ['Axis X:', 'Axis Y:', 'Axis Z:']
            possible_cords = self.__changed_config['max_visible_dim']

        elif self.__cur_used_method == 'PCA':

            # Ak je aktuálne používaná metóda PCA, sú nastavené názvy vstupov a hodnoty vstupov sú nastavené na zákaldne
            # posledne použitých hodnôt zadaných používateľom.
            entry_names = ['PC axis X:', 'PC axis Y:', 'PC axis Z:']
            number_of_pcs = min(self.__changed_config['output_dimension'],
                                self.__changed_config['number_of_samples'])
            # Podľa počtu hlavných komponentov, ktoré je možné zobraziť je do premennej obsahujúcej informáciu o rozsahu
            # možných zadaných hodnôt priradená informácia.
            if number_of_pcs == 0:
                # Ak je počet komponentov 0, znamená to, že nie je možné zobraziť žiadny komponent, preto je do
                # premennej táto informácia priradená.
                cords_label_text = 'No possible PCs:'
            else:

                # Ak je počet komponentov rozdielny od nuly, je do premennej priradená informácia o tom, že je možné
                # voliť z rozsahu 1 až po číslo udávajúce počet komponentov.
                cords_label_text = 'Possible PCs: 1-{}'.format(number_of_pcs)

            # Hodnoty následných troch premenných nazávisia od toho, či vrstva obsahuje alebo neobsahuje feature mapy.
            possible_cords = min(number_of_pcs, self.__changed_config['max_visible_dim'])
            displayed_cords = self.__changed_config['PCA_config']['displayed_cords'].copy()
            displayed_cords = np.array(displayed_cords) + 1
        elif self.__cur_used_method == 't-SNE':

            # Podobne ako pri ostatných metódach je sú aj pre metódu t-SNE preddefinované názvy a hodnoty, ktoré sa majú
            # nastaviť vstupom.
            entry_names = ['t-SNE X:', 't-SNE Y:', 't-SNE Z:']
            possible_cords = self.__changed_config['t_SNE_config']['used_config']['n_components']
            cords_label_text = 'Possible t-SNE components: 0-{}'.format(possible_cords - 1)
            displayed_cords = self.__changed_config['t_SNE_config']['displayed_cords'].copy()

        # Inicializované premenné sú následne zaslané ako argumenty funkcie, ktorá sa stará o zobrazenie názvova hodnôt
        # grafických vstupov pre zadávanie zobrazených súradníc.
        self.set_cords_entries(entry_names, cords_label_text, displayed_cords, possible_cords)

    def set_cords_entries(self, entry_name, cords_label_text, displayed_cords, possible_cords):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastaví názvy a hodnoty vstupov na základe poskytnutých argumentov.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param entry_name:       predstavuje list názvov pre jednotlivé grafické vstuy súradníc.
        :param cords_label_text: obsahuje informáciu o tom, z akého rozashu je možné zvoliť súradnice.
        :param displayed_cords:  list nastavených zobrazovaných hodnôt, ktoré majú byť nastavené ako preddefinované.
        :param possible_cords:   predstavuje počet súradníc, ktoré je možné zobraziť.
        """
        # Najskôr sú všetky súradnicové vstupy skryté. Následne je nastavený text informačnému elementu. Potom v cykle
        # v závislosti od počtu možných súradníc sú zobrazované vstupy a sú im nastavené hodnoty na základe zaslaných
        # argumentov.
        self.hide_entries(self.__cords_entries_list)
        self.__possible_cords_label.setText(str(cords_label_text))
        for i in range(possible_cords):
            cord_entry_rewritable_label = self.__cords_entries_list[i]
            cord_entry_rewritable_label.set_label_name(entry_name[i])
            cord_entry_rewritable_label.set_variable_label(displayed_cords[i])
            cord_entry_rewritable_label.show()

    def set_actual_method_lable(self, method_name):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda nastaví text grafickému prvku, ktorý informuje používateľa o aktuálne používanej metóde
        redukcie priestoru.

        Parameter
        ----------------------------------------------------------------------------------------------------------------
        :param method_name: meno aktuálne zvolenej metódy
        """
        self.__actual_reduc_method.setText(str(f'Actual used: {method_name}'))

    def set_entries_not_marked(self, entries_list):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda označí všetky prvky typu RewritableLabel ako neopozmené, čiže im nastaví defaultnu čiernu farbu a
        odstráni hviezdičku.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param entries_list: list obsahujúci odkazy na RewritableLabel, ktoré treba označiť ako nezmenené.
        """
        for entry in entries_list:
            entry.set_mark_changed(False)

    def hide_option_bar(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda na skrytie obsahu panelu možností. Skryje obaľovací element a následne aj jednotlivé podskupiny.
        :return:
        """
        self.__bar_wrapper.hide()
        self.hide_option_bar_items()

    def hide_option_bar_items(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda ktorá postupne vola funkie na skrytie grafických prvkov v jednotlivých podskupinách.
        """
        self.hide_entries(self.__labels_entries_list)
        self.hide_entries(self.__cords_entries_list)
        self.hide_graph_view_options()
        self.hide_dimension_reduction_options()

    def hide_entries(self, list_of_entries):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda skryje zadanú množinu vstupných grafických prvkov typu RewritableLabel.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param list_of_entries: list odkazov typu RewritableLabel, ktoré majú byť skryté.
        """
        for entry in list_of_entries:
            entry.show_variable_label()
            entry.hide()

    def hide_graph_view_options(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Skryje grafické prvky týkajúce sa zobrazenia a obsahu grafu. Sú skryté len prvky, pri ktorých sa môže stať,
        že nebudú zobrazené. Napríklad checkbox lockview je zobrazený v každom prípade, no chebox color label
        sa zobrazí len v prípade ak boli nahrané labels pre načítané body.
        """
        self.__color_labels_cb.hide()
        self.__3d_view_cb.hide()
        self.__show_polygon_cb.hide()

    def hide_dimension_reduction_options(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda na skrytie grafických prvkov v rámci podskupiny redukcia priestoru.
        """
        self.hide_all_methods_information()

    def hide_all_methods_information(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda skryje obaľovacie elementy, ktoré sú v súčasťou podskupiny ohľadom redukcie priestoru.
        """
        self.__pca_info_gbox.hide()
        self.__t_sne_parameter_gbox.hide()

    def on_method_change(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda sa zavolá po kliknutí na radio button. V rámci tejto metódy sú pre zvolenú metódu zobrazené prípadné
        dodatočné okná.
        """
        # Pre istotu sa otestuje, či je inicializovaná konfiguračná premenná.
        if self.__changed_config is not None:
            # Je získaný názov metódy, ktorá bola zvolená. Do konfiguračnej premennej je zapísaná hodnota zakliknutej
            # metódy pre udržanie informácie o tom, na akú metódu používateľ naposledy klikol a pri prípadnom
            # zobrazení inej vrstvy a následnom zobrazení vrstvy na ktorej bolo kliknuté na niektorú z metód
            # bude táto hodnota už prednastavená.
            method = self.get_checked_method()
            self.__changed_config['config_selected_method'] = method
            if method == 'PCA':

                # Ak je zvolená metóda PCA a je to aj aktuálne používaná metóda pre danú vrstvu, je potrbné zobraziť
                # informácie o výstupe metódy PCA. A skryť iné prípadne zobrazené informácie, konkrétne ak bola
                # predtým zvolená metóda t-SNE, je potrebné skryť kontajner obsahujúci vstupy pre parametre.
                if self.__changed_config['used_method'] == method:
                    self.__pca_info_gbox.show()
                self.__t_sne_parameter_gbox.hide()
            elif method == 't-SNE':

                # Obdobne ako pri metóde PCA je pri zvolení metódy PCA potrebné skryť prípadné informácie o PCA a
                # zobraziť okno s možnosťami zmeny parametrov vstupov pre metódu t-SNE.
                self.__pca_info_gbox.hide()
                self.__t_sne_parameter_gbox.show()
            else:

                # Ak bola zvolená No method, sú skryté všetky dodatočne grafické prvky, ktoré sa zobrazujú pri metóde
                # PCA alebo t-SNE.
                self.__pca_info_gbox.hide()
                self.__t_sne_parameter_gbox.hide()

    def on_color_label_check(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa zavolá v prípade, ak bolo kliknuté na checkbox color_labels. V rámci tejto metódy sa aktualizuje
        konfiguračná premenná, zmenený konfig je na aktívnej vrstve použitý a graf je prekreslený, ak aktívna vrstva
        nejaký graf zobrazuje.
        """
        if self.__changed_config:
            self.__changed_config['color_labels'] = self.__color_labels_cb.isChecked()
            self.__active_layer.use_config()
            self.__active_layer.redraw_graph_if_active()

    def on_lock_view_check(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa zavolá v prípade, ak bolo kliknuté na checkbox lock_view. V rámci tejto metódy sa aktualizuje
        konfiguračná premenná, zmenený konfig je na aktívnej vrstve použitý a graf je prekreslený, ak aktívna vrstva
        nejaký graf zobrazuje.
        """
        if self.__changed_config:
            self.__changed_config['locked_view'] = self.__lock_view_cb.isChecked()
            self.__active_layer.use_config()
            self.__active_layer.redraw_graph_if_active()

    def on_show_polygon_check(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa zavolá v prípade, ak bolo kliknuté na checkbox show_polygon. V rámci tejto metódy sa
        aktualizuje konfiguračná premenná, je nastavená premenná pojednávajuca o tom, či majú byť pre danú
        vrstvu prepočítané hodnoty hrán polygonu, ktorý lemuje zobrazené body. Následne je zmenený konfig
        na aktívnej vrstve použitý a graf je prekreslený, ak aktívna vrstva nejaký graf zobrazuje.
        """
        if self.__changed_config:
            self.__changed_config[
                'show_polygon'] = self.__active_layer.calculate_polygon = self.__show_polygon_cb.isChecked()
            if self.__changed_config['show_polygon']:
                self.__active_layer.set_polygon_cords()
            self.__active_layer.use_config()
            self.__active_layer.redraw_graph_if_active()

    def on_3d_graph_check(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa zavolá v prípade, ak bolo kliknuté na checkbox 3d_view. V rámci tejto metódy sa aktualizuje
        konfiguračná premenná, zmenený konfig je na aktívnej vrstve použitý a graf je prekreslený, ak aktívna vrstva
        nejaký graf zobrazuje.
        """
        if self.__changed_config:
            self.__changed_config['draw_3d'] = self.__3d_view_cb.isChecked()
            self.__active_layer.use_config()
            self.__active_layer.redraw_graph_if_active()

    def validate_label_entry(self, id, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda sa zavolá po tom, čo bolo stlačené tlačidlo enter pri úprave označení osí zobrazovaných na grafe.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param id:    identifikátor vstupu, ktorým sa identifikuje, ktorý vstup má byť upravený. ID nadobúda číslo
                      od 0 po 2 a na základe toho je možné určiť os, pre ktorú sa má nazov nastaviť. Osi idú
                      postupne [Os X, Os Y, Os Z]. Id predstavuje index vliste názvov v konfiguračnej
                      premennej.
        :param value: hodnota odoslaná grafickým prvkom, ktorá ma byť nastavená ako označenie pre os.
        """
        self.__labels_entries_list[id].set_variable_label(value)
        self.__labels_entries_list[id].show_variable_label()
        self.__changed_config['axis_labels'][id] = value
        self.__active_layer.use_config()
        self.__active_layer.redraw_graph_if_active()

    def validate_cord_entry(self, id, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá overuje, či bol vstup zadaný používateľom valídny vzhľadom na aktuálne použitú metódu na redukciu
        priestoru. Metóda sa zavolá po stlačení enter po napísaní požadovanej súradnice na danej osi.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param id:    identifikátor stupu, na ktorom bola zadaná hodnota. Je to index do listu držiaceho odkazy na
                      vstupy ako aj index do listu pre nastavenie súradnice pre určitú os.
        :param value: predstavuje zadanú hodnotu, ktorá sa overuje.

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return       vracia True alebo False podľa toho, či bol vstup korektný alebo nie.
        """
        # Inicializácia premenných, ktoré budú niesť hraničné hodnoty.
        bottom_border = 0
        top_border = 0
        changed_cords = None

        # V závislosti od aktuálne použitej metódy sa menia prípustné hodnoty a aj tvar zadávaných hodnôt, preto je
        # potrebné to rozlíšiť.
        if self.__cur_used_method == 'No method':

            # Ak sa nepoužíva žiadna metóda na redukciu priestoru, je potrebné zistiť, či má vrstva feature mapy. Ak áno
            # vstup bude načítaný v odlišnom tvare ako v ostatných prípadoch.
            if self.__changed_config['has_feature_maps']:

                # V prípade ak má vrstva feature mapy je hodnota testovaná odlišne. Najskôr sa získa referencia na list
                # obsahujúci zobrazované súradnice. Následne sa do premennej priradí aj výstupný tvar vrstvy, ktorý
                # sa použije pri validácií. Na základe premennej obsahujúcej výstupný tvar vrstvy sa nastaví sa
                # do premennej borders načítajú horné hraničné hodnoty. Na prvý index sa načíta hodonota riad-
                # ky, na druhú pozíciu sa načíta horná hranica pre stĺpce. Ďalej sú definované premenné, kto-
                # ré budú obsahovať prípadný chbový text alebo novú hodnotu.
                changed_cords = self.__changed_config['no_method_config']['displayed_cords']
                output_shape = self.__changed_config['output_shape']
                borders = [output_shape[-3] - 1, output_shape[-2] - 1]
                correct_input = []
                output_text = ['', '']

                # Vstup od používateľa je následne rozdelený na základe delimetra čiarky. Je potrebné, aby používateľ
                # dodržal predpísaný tvar vstupu v tvare (riadok, stĺpec). Nadbytočné medzeri by nemali byť problém
                # no dôležité je použitie čiarky. Podľa toho, koľko prvkov bude obsahovať list po rozdelení sa
                # rozhodne, ako sa bude ďalej pokračovať.
                entry_input = value.split(',')
                if len(entry_input) == 2:

                    # Ak bol počet prvkov po rozdelení 2, pokračuje sa testovaním, či sú obe hodnoty číselné a či sa na-
                    # chádzajú v prípustnom intervale. Toto testovanie prebieha v cykle a akonáhle nastane výnimka ale-
                    # bo zadaná hodnota neleží v prípustnom intervale, je do listu pre výpis vložená chybová hláška.
                    # Ak však výnimka nenastane a hodnota leží v požadovanom intervale, je táto hodnota vložená do
                    # listu pre výpis.
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

                    # Na základe hodnôt v liste, ktorý je určený pre výstup je poskladaná výsledná správa.
                    output_msg = f'{output_text[0]}, {output_text[1]}'
                    # Ak sa počet správnych vstupov rovná dvom, všetko prebehlo správne a je možné nastaviť do referen-
                    # cie zobrazovaných súradníc načítané hodnoty. Hodnoty sú zapísané do listu hodnôt na os, ktorá je
                    # určená identifikátorom vstupu.
                    if len(correct_input) == 2:
                        changed_cords[0][id] = correct_input[0]
                        changed_cords[1][id] = correct_input[1]
                        self.__cords_entries_list[id].set_entry_text(output_msg)
                        return_val = True
                    else:
                        return_val = False
                else:

                    # V prípade ak sa počet zadnaých vstupov nerovná dvom je nastavená nasledujúca výstupná správa a
                    # hodnota návratovej premennej na hodnotu False.
                    output_msg = 'err'
                    return_val = False

                # Po vyhodnotení toho, či bol vstup valídny alebo nie sú nastavené zobrazované hodnoty na grafických
                # prvkoch.
                if return_val:
                    # Ak bol vstup valídny je zobrazená nova hodnota súradníc, následne je signalizovaná zmena
                    # zobrazovaných súradníc a je použitá zmenená konfiguračná premenná na aktívnej vrstve.
                    # Taktiež je prekreslený graf.
                    self.__cords_entries_list[id].set_variable_label(output_msg)
                    self.__cords_entries_list[id].show_variable_label()
                    self.__changed_config['cords_changed'] = True
                    self.__active_layer.use_config()
                    self.__active_layer.redraw_graph_if_active()
                else:
                    # Ak vstup nebol správny, je ponechaný zobrazený vstup a je nastavená výstupná správa, ktorá môže
                    # ukazovať, ktoré zo zadaných čísel bolo nesprávne. Ak sa objaví iba jedna chybova hĺaška znamená
                    # to s najväčšou pravdepodobnosťou nesprávny formát vstupu.
                    self.__cords_entries_list[id].set_entry_text(output_msg)
                # Následne je vrátená hodnota o úspešnosti vstupu.
                return return_val
            else:

                # Ak nebola zvolená žiadna metóda redukcie priestoru a vrstva nemá feature mapy, je nastavená spodná a
                # horná hranica pre validáciu prípustných hodnôt. Ak je to možné je vstupná hodnota od používateľa
                # prevedená na int, ak pri pretypovaní dôjde k chybe je na ponechaný zobrazený element na zadanie
                # vstupu a je doň zapísaná hodnota značiaca chybu.
                bottom_border = 0
                top_border = self.__changed_config['output_dimension']
                changed_cords = self.__changed_config['no_method_config']['displayed_cords']
                try:
                    new_value = int(value)
                except ValueError:
                    self.__cords_entries_list[id].set_entry_text('err')
                    return False
        elif self.__cur_used_method == 'PCA':
            # Ak bola je používaná metóda PCA, budú hraničné hodnoty a taktiež aj hodnota uložené do premenných. Pri
            # metóde PCA sú hranice aj hodnota trochu odlišná, lebo komponenty sa zadávajú od čísla 1. Od
            # používateľa príjme hodnotu väčšiu alebo rovnú ako 1, no na pozadí je hodnota odčítaná aby
            # zadaná hodnota mohla predstavovať index do poľa zobrazovaných súradníc. Následne je
            # pretypovaná na int. Ak sa pri pretypovaní neobjaví výnimka, pokraćuje sa ďalej, ak
            # áno je nastavená hodnota err ako text do zobrazovaného vstupu.
            bottom_border = 1
            top_border = min(self.__changed_config['output_dimension'],
                             self.__changed_config['number_of_samples']) + 1
            changed_cords = self.__changed_config['PCA_config']['displayed_cords']
            try:
                new_value = int(value) - 1
            except ValueError:
                self.__cords_entries_list[id].set_entry_text('err')
                return False
        elif self.__cur_used_method == 't-SNE':
            # Ak sa použije metóda t-SNE, sú podľa konfigurácie nastavené hraničné hodnoty. Zadaná hodnota je následne
            # pretypovaná na typ int. Ak sa pri pretypovaní objaví chyba, je do vstupu písaná hodnota err.
            bottom_border = 0
            top_border = self.__changed_config['t_SNE_config']['used_config']['n_components']
            changed_cords = self.__changed_config['t_SNE_config']['displayed_cords']
            try:
                new_value = int(value)
            except ValueError:
                self.__cords_entries_list[id].set_entry_text('err')
                return False

        # Na základe nastavených hodnôt a hraníc sa vyhodnotí, či je vstup valídny. Ak áno je nová hodnota priradená
        # do konfiguračnej premennej a sú nastavené aj novo zobrazované hodnoty. Ak bol vstup valídny je signali-
        # zovaná zmena vstupu a následne je konfiguračná premenná pre aktívnu vrstvu použítá. Na záver je
        # prekreslený graf.
        try:
            if not (bottom_border <= int(value) < top_border):
                self.__cords_entries_list[id].set_entry_text('err')
                return False

            self.__cords_entries_list[id].set_variable_label(value)
            self.__cords_entries_list[id].show_variable_label()
            changed_cords[id] = int(new_value)
            self.__changed_config['cords_changed'] = True
            self.__active_layer.use_config()
            self.__active_layer.redraw_graph_if_active()
            return True
        except ValueError:
            self.__cords_entries_list[id].set_entry_text('err')
            return False

    def validate_t_sne_entry(self, id, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Validuje vstup pre parametre metódy t-SNE.

        Parmetre
        ----------------------------------------------------------------------------------------------------------------
        :param id:    identifikátor vstupu.
        :param value: hodnota zadaná na vstupe

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return: navracia True alebo False na základe toho, či bol zadaný vstup valídny.
        """
        # Všetko prebieha v try catch blocku. Identifikator vstupu je aj kľúčom k odkazu na grafický prvok pre vstup
        # od užívateľa ako aj kľúč v rámci konfiguračného súboru, kde sú uložené prípustné hranice pre parameter
        # a taktiež a typová hodnota parametra.
        try:
            if self.__changed_config is not None:

                # Na základe id je do premennej test_tuple priradený tuple obsahujúci na prvej pozícií spodnú hranicu
                # prípustného intervalu, na druhej pozicií prípustný typ zadanej hodnoty a na tretej pozicií hornú
                # hranicu intervalu. Zadaná hodnota je najskôr pretypovaná na prípustný typ. Ak nedôjde pri
                # pretypovaní k chybe, je hodnota porovnaná s hraničnými hodnotami. Ak zadaná hodnota leží
                # v prípustnom intervale, je hodnota priradená do konfiguračnej premennej. AK nie, je
                # na vstupe vypísaná chybová hláška. Ak je nová hodnota odlišná od posledne použitej
                # hodnoty, označí sa grafický prvok ako zmenený, to znamená, že sa zmení jeho farba
                # na červenú a k jeho názvu je pridaná hviezidčka.
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
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda vracia na základe zakliknutého radio buttonu názov metódy.

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return: string nesúci názov použitej metódy
        """
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
