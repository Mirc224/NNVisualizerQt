from PyQt5 import QtCore, Qt
from PyQt5 import QtGui
from PyQt5.QtWidgets import *


class ClickLabel(QLabel):
    """
    Popis
    --------------------------------------------------------------------------------------------------------------------
    Subtrieda triedy QLabel, umožňujúca vyslať signál, keď dôjde ku kliknutiu na label.
    """
    clicked = QtCore.pyqtSignal()

    def mousePressEvent(self, event):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Prepísanie eventu mousePressEvent. Metóda sa zavolá pri výskyte eventu klinutia na label. Inštancia vyšle signal
        a taktiež vykoná aj defaultnu akciu spojenú s týmto eventom.

        Paremetre
        ----------------------------------------------------------------------------------------------------------------
        :param event: obsahuje informácie o vyskytnutej udalosti.
        """
        self.clicked.emit()
        QLabel.mousePressEvent(self, event)


class BackEntry(QLineEdit):
    """
    Popis
    --------------------------------------------------------------------------------------------------------------------
    Subtrieda triedy QLineEdit, umožňujúca vyslať signál, keď pri aktivovanom QLineEidt elemente je stlačené tlačidlo
    escape.
    """

    escPress = QtCore.pyqtSignal()

    def keyPressEvent(self, event):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Prepísanie metódy keyPressEvent, ktorá je zavolaná v prípade, že bolo stlačené tlačidlo esc, keď bol aktivovaný
        element QLineEdit. Je zachovaná aj defaultna funkcia elementu.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param event: obsahuje inofrmácie o vysktnutej udalosti.
        """
        if event.key() == QtCore.Qt.Key_Escape:
            self.escPress.emit()
        QLineEdit.keyPressEvent(self, event)


class RewritableLabel(QWidget):
    def __init__(self, element_id=None, label_text='', variable_text='', enter_command=None, entry_width=50,
                 *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Trieda predstavuje grafický komponent s označeným vstupom umožňujúci prepisovať hodnoty elementu QLabel.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__element_id:    identifikátor inštancie
        :var self.__name_variable: obsahuje hodnotu, ktorá má byť zobrazená ako označenie vstupu.
        :var self.__mark_changed:  atribút nesie informáciu o tom, či je inštancia označená ako zmenená. To je použité
                                   pri grafickom odlíšení inštancií.
        :var self.__name_label:    inštancia triedy QLabel, ktorá vypisuje označenie RewritableLabel.
        :var self.__enter_func:    referencia na funkciu, ktorá sa má vykonať v prípade, že bude aktivovaný element 
                                   QLineEdit a bude stlačený enter.
        :var self.__var_entry:     inštancia triedy BackEntry, ktorá bude preberať vstup zadaný používateľom.
        :var self.__var_label:     inštancia triedy ClickLabel, ktorá po validácii zobrazuje obsah, ktorý používateľ
                                   zadal na vstupe.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param element_id:    identifikátor inštancie.
        :param label_text:    text, obsahujúci označenie vstupu.
        :param variable_text: text, ktorým bude inicializovana QLabel zobrazujúca prepisovateľnú hodnotu.
        :param enter_command: referencia na funkciu.
        :param entry_width:   šírka vstupného elementu
        :param args:          dodatočné argumenty.
        :param kwargs:        keyword argumenty.
        """
        super(RewritableLabel, self).__init__(*args, **kwargs)
        # Sú inicializované základné atribúty.
        self.__id = element_id
        self.__mark_changed = False
        self.__enter_func = enter_command
        # K názvu je pirávané jedno voľné miesto, ktoré bude v prípade, že je inštancia označná za zmenenú obsahovať *.
        self.__name_variable = label_text + ' '

        # Vytvorenie grafických elementov.
        self.__name_label = QLabel(self.__name_variable, self)
        self.__var_entry = BackEntry(self)
        self.__var_label = ClickLabel(variable_text, self)

        # Incializácia grafických komponentov triedy.
        self.initialize_ui(entry_width)

    def initialize_ui(self, entry_width):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        V metóde sú inicializované grafické prvky, sú rozmiestnené v obľovacom prvku a sú im nastavené niektoré grafické
        vlastnosti.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param entry_width: obsahuje požadovanú šírku komponent vstupu.
        """
        self.__var_entry.setFixedWidth(entry_width)
        self.__var_entry.escPress.connect(self.esc_pressed)
        self.__var_entry.returnPressed.connect(self.enter_pressed)
        self.__var_entry.hide()
        self.__var_label.clicked.connect(self.show_entry)

        layout = QHBoxLayout(self)

        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.__name_label, alignment=QtCore.Qt.AlignRight)
        layout.addWidget(self.__var_label, alignment=QtCore.Qt.AlignLeft)
        layout.addWidget(self.__var_entry, alignment=QtCore.Qt.AlignLeft)
        self.setMaximumHeight(50)

    def enter_pressed(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda sa zavolá po výskyte udalosti stlačenia enter, v prípade že je aktivovaný komponent vstupu. Ak bola pos-
        kytnutá funkcia, ktorá sa má pri tejto udalosti zavolať, je táto funkcia zavolaná spolu s argumentami, ktorými
        sú identifikátor inštancie triedy RewritableLabel a obsah komponentu vstupu, ktorý sa práve nachádza v text-
        ovej oblasti.
        """
        if self.__enter_func is not None:
            self.__enter_func(self.__id, self.__var_entry.text())

    def show_entry(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zobrazenie grafického komponentu vstupu, do ktorého je vložená hodnota textu z grafického elementu ClickLabel.
        Komponent vstupu je zobrazený, zatiaľ čo komonent ClickLabel je skrytý.
        """
        self.__var_entry.setText(self.__var_label.text())
        self.__var_label.hide()
        self.__var_entry.show()

    def set_label_name(self, name):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie označenia vstupu.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param name: označenie, ktoré má byť použité.
        """
        self.__name_variable = name + ' '
        self.__name_label.setText(str(self.__name_variable))

    def set_entry_text(self, text=''):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie textu, ktorý má byť zobrazený ako preddefinovaná hodnota po objavení vstupu.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param text: text, ktorý má byť zobrazený v komponente vstupu.
        """
        self.__var_entry.setText(str(text))

    def set_variable_label(self, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie textu zobrazovaného kompoentom ClickLabel, ktoré úlohou je reprezentácia nastavenej hodnotu na vstupe.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param value: text, ktorý má byť nastavený na zobrazenie.
        """
        self.__var_label.setText(str(value))

    def show_variable_label(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zobrazí komponent ClickLabel a skryje komponent vstupu.
        """
        self.__var_entry.hide()
        self.__var_label.show()

    def get_variable_value(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vráti text, ktorý je nastavený na komponente ClickLabel.
        """
        return self.__var_label.text()

    def set_mark_changed(self, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Je nastavená hodnota atribútu mark_changed. Podľa tejto hodnoty sa nastaví vzhľad a zobrazovaný text. Ak bola
        zadaná hodnota True, je inštancia označená ako changed a textu označenia vstupu je priadaná hviezidčka a je
        zmenená farba na červenú. Ak bola zadaná hodnota False, má zobrazený text čiernu farbu.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param value: nová hodnota atribútu.
        """
        self.__mark_changed = value
        if self.__mark_changed:
            # Z textu v premennej name_variable sú vybraté znaky od prvého až po predposledný, pretože tým je medzera.
            # Prázdny znak je použitý, aby ostala konzistentná dĺžka komponentu QLabel. V tomto prípade je medzera
            # nahradená hviezdičkou.
            self.__name_label.setText("<font color='red'>{}</font>".format(str(self.__name_variable[:-1] + "*")))
        else:
            self.__name_label.setText("<font color='black'>{}</font>".format(str(self.__name_variable)))

    def esc_pressed(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zobrazí komponent ClickLabel obsahujúci text zadaný na vstupe. Je to metóda, ktorá je stlačená keď je pri akti-
        vovanom vstupe stlačené tlačidlo escape.
        :return:
        """
        if self.__var_entry.isVisible():
            self.show_variable_label()


class RemovingCombobox(QWidget):
    def __init__(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Komponent obsahuje combobox a tlačidlo. V comobobx obsahuje zoradenú množinu prvkov, ktoré je možné vybrať. Prv-
        ky je možné odstraňovať z comboboxu, ak sa nachádzajú v inicializovanej množine prípustných prvkov. Prvky
        je možné aj opätovne zobraziť, ak sa nachádzajú v inicializovanej množine prvkov.
        
        Atribúty
        ---------------------------------------------------------------------------------------------------------------- 
        :var self.__default_text:   text, ktorý je zobrazený pri vytvorení inštancie, alebo zvolení a odstránení prvku.
        :var self.__read_only:      bool hodnota, nesúca informáciu o tom, či je možné do ComboBoxu písať, alebo je
                                    možný iba výber z definovaných možností.
        :var self.__next_spec_ID:   hodnota nasledujúceho špeciálneho ID prvku.
        :var self.__all_values:     slovník obshaujúci všetky hodnoty v prípustných prvkov. Názov prvku v comboboxe je
                                    kľúčom do tohto slovníka, pomocou ktorého je možné získať ID prvku.
        :var self.__already_slct:   zoznam hodnôt Comboboxu, ktoré už boli zvolené a vyradené zo zobrazených prvkov.
        :var self.__ordered_values: zoznam obsahuje označenia prvkov v Comboboxe v poradí, v ktorom majú byť zobrazené.
        :var self.__backw_values:   spätný slovník, kde na základe identifikátora je možné získať názov prvku Comboboxu,
                                    ktorému prislúcha.
        :var self.__combobox:       inštancia grafického komponentu Combobox. Zobrazuje poskytnuté prvky.
        :var self.__add_btn:        inštancia grafického komponentu PushButton. Pomocou neho je vyvolaná funkcia nad
                                    zvoleným prvkom.
        :var self.__command:        obsahuje odkaz na funkciu, ktorá sa má zavolať po stlačení tlačidla.
        """

        super(RemovingCombobox, self).__init__(*args, **kwargs)

        self.__read_only = ''
        self.__command = None
        self.__default_text = ''
        self.__next_spec_ID = None

        self.__all_values = {}
        self.__backw_values = {}
        self.__already_slct = []
        self.__ordered_values = []

        self.__combobox = QComboBox(self)
        self.__add_btn = QPushButton('Add', self)

        self.initialize_ui()

    def initialize_ui(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializácia grafických komponentov.
        """
        layout = QVBoxLayout(self)
        self.__combobox.setInsertPolicy(QComboBox.NoInsert)
        self.__combobox.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.__combobox.view().setUniformItemSizes(True)
        layout.addWidget(self.__combobox, alignment=QtCore.Qt.AlignBottom)
        self.__add_btn.clicked.connect(self.show_selected)
        layout.addWidget(self.__add_btn, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

    def initialize(self, item_list, command=None, button_text: str = 'Add', read_only: bool = True,
                   default_text: str = ''):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializovanie atribútov.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param item_list:    zoznam prvkov, ktoré majú byť v Comboboxe zobrazené.
        :param command:      referencia na funkciu, ktorá sa má vykonať po kliknutí na tlačidlo.
        :param button_text:  text zobrazený na tlačidle.
        :param read_only:    bool hodnota, hovoriaca o tom, či má byť Combobox read only, alebo doň bude možné písať.
        :param default_text: text, ktorý má byť zobrazený pri vytvorení, prípadne odobratí prvku zo zobrazovaných
                             prvkov.
        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return vracia zoznam unikátnych názvov, vytvorený na základe poskyntutého zoznamu prvkov.
        """
        # Hodnoty slovníkov a zoznamov sú vyčistené.
        self.__all_values = {}
        self.__backw_values = {}
        self.__already_slct = []
        self.__ordered_values = []

        self.__command = command
        self.__next_spec_ID = 1

        self.__combobox.clear()

        self.__add_btn.setText(str(button_text))
        self.__read_only = read_only
        self.__default_text = default_text
        if self.__read_only:
            self.__combobox.setEditable(False)
        else:
            self.__combobox.setEditable(True)

        # Ak je Comobobx typu read only a bol poskytnutý nejaký default text, ktorý má byť zobrazovaný, je pridaný do
        # zoznamu zobrazovaných prvkov ako prvý.
        if self.__read_only and self.__default_text != '':
            self.__ordered_values.append(default_text)

        tmp_set = set()
        prev_set_len = len(tmp_set)

        # Hodnoty z poskytnutého zoznamu sú postupne prechádzané a testované, či sú v rámci zobrazovaných prvkov jedi-
        # nečné.
        for i, item_name in enumerate(item_list):
            # Prvok je pridaný do pomocnej štruktúry set. Ak sa počet prvkov v množine nie je odlišný od predhcádzajú-
            # ceho počtu prvkov znamená to, že prvok sa už v zozname nachádza, preto je preň potrebné vytvoriť uni-
            # kátne meno. Po vytvorení unikátneho mena, je toto meno vložené do slovníka všetkých hodnôt, kde je
            # mu priradené ID, do slovníka spätných hodnôt kde je na zákade ID je vložené unikátne meno.
            tmp_set.add(item_name)
            if len(tmp_set) == prev_set_len:
                item_name = self.get_unique_name(item_name)
                tmp_set.add(item_name)
            prev_set_len = len(tmp_set)
            self.__all_values[item_name] = i
            self.__ordered_values.append(item_name)
            self.__backw_values[i] = item_name

        # Nasleduje updatovanie listu zobrazených prvkov a prispôsobenie veľkosti tlačidla.
        self.update_list()
        self.__add_btn.adjustSize()

        # Podľa toho, či je combobox nastavený na read only a či je nastavený defaultný text, je vrátený zoznam unikát-
        # nych prvkov.
        if self.__read_only and self.__default_text != '':
            # Ak bol nastavený defaultný text, nachádza sa v zozname na prvom mieste place holder, preto je ho potrebné
            # preskočiť a pokračovať začať až od indexu 1.
            return self.__ordered_values[1:].copy()
        else:
            return self.__ordered_values.copy()

    def get_unique_name(self, item_name):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Otestuje, či sa poskytnuté meno nenachádza už v zozname unikátnych prvkov, ak áno je vytvorené unikátne meno
        podobné pôvodnému.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param item_name: prvok, ktorého unikátnosť chceme cestovať.

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return je vrátený unikátny prvok.
        """
        if item_name in self.__all_values:
            new_name = item_name + ' ({})'
            number_of_copy = 1
            while True:
                if new_name.format(number_of_copy) in self.__all_values:
                    number_of_copy += 1
                    continue
                else:
                    item_name = new_name.format(number_of_copy)
                    break
        return item_name

    def get_list_of_visible(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vytvorí a vráti list prvkov, ktoré majú byť zobrazené v Comboboxe.
        """
        return [layer_name for layer_name in self.__ordered_values if layer_name not in self.__already_slct]

    def show_selected(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Pošle do priradenej funkcie hodnoty súvisiace so zvoleným prvkom. Pred zavolaním funkcie a skontroluje, či sa
        zvolený prvok nachádza v zozname prípustných prvkov.
        """
        selected = self.__combobox.currentText()
        if self.__command is not None and selected in self.__all_values:
            self.__command((self.__all_values[selected], selected))

    def update_list(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Aktualizácia zobrazených prvkov v comboboxe a prípadne nastavenie preddefinovaného textu.
        """
        visible_list = self.get_list_of_visible()
        self.__combobox.clear()
        self.__combobox.addItems(visible_list)
        if self.__read_only:
            if self.__default_text != '':
                self.__combobox.setCurrentIndex(0)
        else:
            self.__combobox.setCurrentText(self.__default_text)

    def hide_item(self, item_name):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Skryje prvok zo zoznamu zobrazených prvkov comboboxu.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param item_name: meno prvku, ktoré má byť zo zonamu vyradené.
        """
        if item_name in self.__all_values:
            self.__already_slct.append(item_name)
            self.update_list()

    def show_item(self, item_name):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Pridá prvok do zoznamu zobrazovaných prvkov comboboxu, ak je tento prvok v defunovanom zozname prípustných
        prvkov.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param item_name: názov prvku, ktorý má byť zobrazený v zozname.
        """
        if item_name in self.__already_slct:
            if self.__read_only:
                if self.__default_text != item_name:
                    self.__already_slct.remove(item_name)
                    self.update_list()
            else:
                self.__already_slct.remove(item_name)
                self.update_list()

    def add_special(self, item_name):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Pridá do zoznamu špeciálny prvok. Tento je pri zobrazení vložený na vrch zoznamu zobrazovaných prvkov, prípadne
        na druhé miesto, ak je nastavený preddefinovaný text pri read only Comboboxoch.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param item_name: meno prvku, ktorý má byť zobrazený.

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return unikátne meno špeciálneho prvku.
        """
        new_list = []
        starting_index = 0
        # Podľa toho, či je preddefinovaný zobrazovaný text je nastavený index v zozname prvkov, na ktorý byť špeciálny
        # prvok vložený.
        if self.__read_only and self.__default_text != '':
            new_list.append(self.__default_text)
            starting_index += 1

        # Následne je získané unikátne meno pre špecialneho prvku. Podľa tohto mena je prvok uložený do zoznamov.
        item_name = self.get_unique_name(item_name)
        new_list.append(item_name)
        new_list.extend(self.__ordered_values[starting_index:])
        self.__ordered_values = new_list
        self.__all_values[item_name] = -self.__next_spec_ID
        self.__next_spec_ID += 1
        self.update_list()
        return item_name

    def clear(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vymaže grafické prvky a uvoľní pridelené prostridky.
        """
        self.__add_btn.destroy()
        self.__combobox.destroy()
        self.__add_btn.deleteLater()
        self.__combobox.deleteLater()
        self.deleteLater()
        self.__already_slct = None
        self.__backw_values = None
        self.__add_btn = None
        self.__all_values = None
        self.__combobox = None
        self.__command = None


class FloatSlider(QSlider):
    onResize = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Ide o odvodenú triedu od triedy QSlider, ktorá umožňuje pracovať s desatinnými hodnotami.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self._max_int:     obsahuje maximálnu hondot int.
        :var self._min_value:   minimálna hodnota na slideri. Používa sa pri výpočte a prevode an desatinné číslo.
        :var self._max_value:   maximálna hodnota na slideri. Používa sa pri výpočte a prevode an desatinné číslo.
        """
        super(FloatSlider, self).__init__(*args, **kwargs)
        self._max_int = 10 ** 5
        super().setMinimum(0)
        super().setMaximum(self._max_int)
        self._min_value = 0
        self._max_value = 1

    def value(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Prepisuje funkciu zákaldnej triedy QWidget, ktorá sa poskytne aktuálnu hodnotu na slideri. Podľa nastavení sa
        vykoná prevod na desatinné číslo.
        """
        return float(super().value()) / self._max_int * (self._max_value - self._min_value) + self._min_value

    def setValue(self, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastaví pozíciu a hondotu slidera po vykonaní prepočtu.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param value: požadovaná hodnota, ktorá má byť nastavená.
        """
        super().setValue(int((value - self._min_value) / (self._max_value - self._min_value) * self._max_int))

    def setMinimum(self, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie minimálnej hodnoty, ktorá môže byť na slideri nastavená.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param value: požadovaná hodnota, ktorá má byť nastavená ako minimum.
        """
        if value > self._max_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._min_value = value
        self.setValue(self.value())

    def setMaximum(self, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie maximálnej hodnoty, ktorá môže byť na slideri nastavená.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param value: požadovaná hodnota, ktorá má byť nastavená ako maximum.
        """
        if value < self._min_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._max_value = value
        self.setValue(self.value())

    def minimum(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vráti minimálnu hodnotu, ktorú je možné na slideri nastaviť.
        """
        return self._min_value

    def maximum(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vráti maximálnu hodnotu, ktorú je možné na slideri nastaviť.
        """
        return self._max_value

    def resizeEvent(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Prepísaná funkcia v základnej triede, ktorá zavolá pôvodnú funkciu pre udalosť zmeny veľkosti a vyšle signál o
        zmene veľkosti.
        """
        super().resizeEvent(*args, **kwargs)
        self.onResize.emit()

    def wheelEvent(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Prepisuje pôvodnú funkciu wheelEvent, čím zabraňuje zmenám nastavenej hodnoty pomocou koliečka myši.
        """
        pass


class DisplaySlider(QWidget):
    def __init__(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Pomenovaný slider, ktorý zobrazuje nad bežcom vypisuje nastavenú hodnotu a po kliknutí na hodnotu je ju možné
        priamo zadať z klávesnice. Obsahuje aj tlačidlo, na ktoré je možné nastaviť funkciu, ktorá sa má stlačení
        vykonať.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__id:         atribút, ktorý umožňuje nastavenia identifikátora slidera.
        :var self.__wrapper_gb: obaľovací element. Je mu nastavené meno slidera.
        :var self._slider:      slider umožňujúci nastavenie desatinných čísel.
        :var self.__ent_lbl_gb: obaľuje elementy vstupu a labelu zobrazujúceho nastavenú hodnotu.
        """
        super(DisplaySlider, self).__init__(*args, **kwargs)
        self.__id = None
        self.__wrapper_gb = QGroupBox()
        self._slider = FloatSlider(QtCore.Qt.Horizontal)
        self.__ent_lbl_gb = QFrame(self.__wrapper_gb)
        self.__hide_btn = QPushButton()
        self.__value_label = ClickLabel(self.__ent_lbl_gb)
        self.__value_entry = BackEntry(self.__ent_lbl_gb)
        self.initialze_ui()

    def initialze_ui(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Incializácia grafických komponentov a ich rozmiestnenie.
        """
        v_layout = QVBoxLayout(self)
        v_layout.setSpacing(0)

        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.addWidget(self.__wrapper_gb)
        self.__wrapper_gb.setTitle('Nadpis')
        display_layout = QVBoxLayout()

        display_layout.addWidget(self.__value_label)
        display_layout.addWidget(self.__value_entry)
        self.__ent_lbl_gb.setLayout(display_layout)

        wrapper_layout = QHBoxLayout()
        self.__wrapper_gb.setLayout(wrapper_layout)
        wrapper_layout.setSpacing(10)
        wrapper_layout.setContentsMargins(10, 15, 5, 5)

        wrapper_layout.addWidget(self._slider)

        self.__hide_btn.setText('X')
        self.__hide_btn.setStyleSheet('QPushButton { color: red; }')
        self.__hide_btn.setMaximumWidth(20)
        wrapper_layout.addWidget(self.__hide_btn, alignment=QtCore.Qt.AlignRight)

        self.__value_label.setText(str(self.get_formated_value()))
        self.__value_label.adjustSize()
        self.__value_label.clicked.connect(self.show_entry)

        self.__value_entry.hide()
        self.__value_entry.setFixedWidth(30)
        self.__value_entry.escPress.connect(self.show_label)
        self.__value_entry.returnPressed.connect(self.return_pressed)

        self._slider.setMaximum(50)
        self._slider.setMinimum(10)
        self._slider.valueChanged.connect(self.on_value_change)
        self._slider.onResize.connect(self.display_value)

    def initialize(self, slider_id, minimum=0, maximum=100, slider_name='Slider', on_change_command=None,
                   hide_command=None, value=None):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializácia atribútov na zákalde poskytnutých parametrov.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param slider_id:          identifikátor slidera, ktorý sa bude vracať pri niektorých funkcíach.
        :param minimum:            minimálna hodnota, ktorú je bude na slideri možné nastaviť.
        :param maximum:            maximálna hodnota, ktorú je bude na slideri možné nastaviť.
        :param slider_name:        meno slidera, ktoré sa bude zobrazovať v ľavom hornom okraji.
        :param on_change_command:  referencia na funkciu, ktorá sa má zavolať pri zmene veľkosti komponentu.
        :param hide_command:       odkaz na funkciu, ktorá bude zavolaná pri stlačení tlačidla.
        :param value:              hodnota, ktorá má byť na slideri nastavená
        """
        self.__id = slider_id
        self._slider.setMinimum(minimum)
        self._slider.setMaximum(maximum)
        self.__wrapper_gb.setTitle(slider_name)
        if value is not None:
            self._slider.setValue(value)
        if on_change_command is not None:
            self._slider.valueChanged.connect(lambda: on_change_command(self._slider.value()))
        if hide_command is not None:
            self.__hide_btn.clicked.connect(lambda: hide_command(slider_id))

    def on_value_change(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Prepísanie funkcie, ktorá sa má vykonať pri zmene hodnoty slidera.
        """
        self.display_value()

    def get_formated_value(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vráti naformatovanú hodnotu nastaveného čísla na slideri.
        """
        new_value = str('{0:.2f}'.format(round(self._slider.value(), 2)))
        return new_value.rstrip('0').rstrip('.') if '.' in new_value else new_value

    def display_value(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nad bežcom zobrazí práve nastavenú hodnotu.
        """
        self.__value_label.setText(str(self.get_formated_value()))
        self.__value_label.adjustSize()
        self.__ent_lbl_gb.adjustSize()
        new_x = self.convert_to_pixels(self._slider.value())
        self.__ent_lbl_gb.move(int(new_x), self._slider.y() - 30)

    def show_entry(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zobrazí vstup pre nastavenie hodnoty pomocou klávesnice. Je zavolaná po kliknutí na aktuálne zobrazovanú hodno-
        tu.
        """
        self.__value_label.hide()
        self.__value_entry.setText(str(self.__value_label.text()))
        self.__value_entry.show()
        self.__value_entry.adjustSize()
        self.__value_entry.setFixedHeight(self.__value_label.height())
        self.__ent_lbl_gb.adjustSize()
        self.display_value()

    def show_label(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Skryje komponent na zadávanie a zobrazí komponent zodpovedný za výpis aktuálne nastavenej hodnoty.
        """
        self.__value_entry.hide()
        self.__value_label.show()
        self.display_value()

    def convert_to_pixels(self, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vráti hodnotu v pixeloch, kde by sa mal nachádzať obaľovací element vstupu a zobrazovacieho prvku pri konrétnej
        nastavenej hodnote na slideri.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param value: hodnota, na ktorej je slider nastavený.
        """
        hodnota = ((value - self._slider.minimum()) / (self._slider.maximum() - self._slider.minimum()))
        return hodnota * (self._slider.width() - self.__ent_lbl_gb.width() / 2)

    def validate_entry(self, ):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Overí, či hodnota zadaná na vstupe je valídna.
        """
        try:
            if not (self._slider.minimum() <= float(self.__value_entry.text()) <= self._slider.maximum()):
                return False
            self._slider.setValue(float(self.__value_entry.text()))
            return True
        except ValueError:
            return False

    def esc_pressed(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Skryje vstup a zobrazí element vypisujúci aktuálne nastavenú hodnotu slidera. Zavolá sa, ak je aktivovaný vstup
        a je stlačená klávesa escape.
        """
        if self.__var_entry.isVisible():
            self.show_variable_label()

    def return_pressed(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zistí, či je zadaný vstup valídny a podľa toho buď zobrazí element zodpovedný za vypisovanie hodnoty alebo nas-
        taví chybovú skratku, ktorá sa použíateľovi zobrazí.
        """
        if self.validate_entry():
            self.show_label()
        else:
            self.__value_entry.setText('err')

    def clear(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vyčistenie inštancie a uvoľnenie pridelených prostriedkov.
        """
        self.__hide_btn.deleteLater()
        self.__hide_btn.clicked.disconnect()
        self.__hide_btn = None
        self._slider.deleteLater()
        self._slider.valueChanged.disconnect()
        self._slider = None
        self.__wrapper_gb.deleteLater()
        self.__wrapper_gb = None
        self.deleteLater()


class VariableDisplaySlider(DisplaySlider):
    def __init__(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Odvodená trieda od triedy DisplaySlider, ktorá k jej funkciám pridáva možnosť nastaviť premennú v mutable obje-
        kte akým je napríklad pole, ktorá bude priamo mená pri nastavení hodnoty slidera.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__variable_list: referencia na mutable objekt, v ktorom má byť nastavovaná hodnota.
        :var self.__index:         index na menenú hodnotu v mutable objekte.
        """
        super(VariableDisplaySlider, self).__init__(*args, **kwargs)
        self.__variable_list = None
        self.__index = None

    def initialize(self, slider_id, minimum=0, maximum=100, slider_name='Slider', on_change_command=None,
                  hide_command=None, variable_list=None, index=None):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Incicializácia atribútov.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param slider_id:          identifikátor slidera, ktorý sa bude vracať pri niektorých funkcíach.
        :param minimum:            minimálna hodnota, ktorú je bude na slideri možné nastaviť.
        :param maximum:            maximálna hodnota, ktorú je bude na slideri možné nastaviť.
        :param slider_name:        meno slidera, ktoré sa bude zobrazovať v ľavom hornom okraji.
        :param on_change_command:  referencia na funkciu, ktorá sa má zavolať pri zmene veľkosti komponentu.
        :param hide_command:       odkaz na funkciu, ktorá bude zavolaná pri stlačení tlačidla.
        :param variable_list:      referencia na mutable objekt.
        :param index:              index menenej hodnoty v mutable objekte.

        """
        value = None
        if variable_list is not None and index is not None:
            self.set_variable(variable_list, index)
            value = variable_list[index]
        super().initialize(slider_id, minimum, maximum, slider_name, on_change_command, hide_command, value)

    def set_variable(self, var_list, index):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenia mutable objektu a indexu k príslušnej hodnote, ktorá má byť menená.
        :param var_list:
        :param index:
        :return:
        """
        self.__variable_list = var_list
        self.__index = index
        self._slider.setValue(self.__variable_list[self.__index])

    def on_value_change(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Pri zmene slidera nastaví hodnotu v mutable objekte na hodnotu slidera.
        """
        if self.__variable_list is not None:
            self.__variable_list[self.__index] = self._slider.value()
        super().on_value_change()


class CustomMessageBox(QMessageBox):
    def __init__(self, window_title='', text='', yes_button='Yes', no_buton='No', cancel='Cancel', informative_text='',
                 *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vytvorí message box s troma tlačidami, naplnený poskytnutými parametrami.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param window_title:     nadpis message boxu.
        :param text:             text zobrazený vo vytvorenom okne.
        :param yes_button:       nápis na yes tlačidle.
        :param no_buton:         nápis na no tlačidle.
        :param cancel:           nápis na cancel tlačidle.
        :param informative_text: informatívny text.
        """
        super(CustomMessageBox, self).__init__(*args, **kwargs)
        self.setWindowTitle(window_title)
        self.setText(text)
        self.addButton(yes_button, QMessageBox.YesRole)
        self.addButton(no_buton, QMessageBox.NoRole)
        self.addButton(cancel, QMessageBox.RejectRole)
        self.setInformativeText(informative_text)