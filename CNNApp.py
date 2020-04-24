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
        graph_page = GraphPage(self)
        self.setCentralWidget(graph_page)


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

            # Ak bol nejaký model už načítaný, objaví sa dialógové okno, ktoré umožní používateľovi zvoliť miesto a 
            # názov pod ktorým chce model uložiť. Dialógové okno sa otvára v priečinku, z ktorého bol model
            # načítaný a ako preddefinovaný názov je nastavený názov načítaného modelu.
            file_path = QFileDialog.getSaveFileName(self, 'Save Keras model',
                                                    ntpath.join(self.__file_path,self.__file_name),
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


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


sys.excepthook = except_hook
app = QApplication(sys.argv)
window = VisualizationApp()
window.showMaximized()

app.exec_()
