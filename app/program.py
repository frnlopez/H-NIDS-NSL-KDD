import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import sys
import joblib

# Raíz del repositorio: este fichero vive en <raíz>/app/program.py. El dataset
# se lee de <raíz>/dataset y las salidas van a <raíz>/Resultados.
RAIZ_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# NO se silencian los warnings a nivel de módulo: un
# `warnings.filterwarnings('ignore')` global ocultaría avisos útiles
# (convergencia de sklearn, deprecaciones, divisiones por cero de numpy). Si un
# aviso molesta, se acota al punto concreto que lo emite, no a todo el proceso.

# Forzar salida UTF-8 en consolas Windows: sin esto, los prints con emojis
# cascan con el códec cp1252 salvo que exista PYTHONIOENCODING=utf-8.
if sys.stdout.encoding and sys.stdout.encoding.lower().replace('-', '') != 'utf8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # stdout no reconfigurable (p. ej. redirigido a un objeto sin reconfigure):
        # se continúa sin forzar el encoding
        pass

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")

# ─────────────────────────────────────────────────────────────────────────────
# FUENTE CANÓNICA de las columnas categóricas del NSL-KDD y de las columnas que
# no son características. Están a nivel de módulo —y no solo dentro de
# NSLKDDPreprocessor— para que validacion.py las importe en vez de copiarlas:
# duplicadas por copia, un cambio aquí rompería en silencio la reproducción del
# one-hot en medir_vocabulario_onehot().
# ─────────────────────────────────────────────────────────────────────────────
COLUMNAS_CATEGORICAS = ['protocol_type', 'service', 'flag']
COLUMNAS_NO_CARACTERISTICA = ['attack', 'level', 'attack_category']


class NSLKDDPreprocessor:
    """
    Clase para el preprocesamiento completo del dataset NSL-KDD
    con división especializada para modelos de anomalías y firmas
    """
    
    def __init__(self):
        # Nombres de las 41 características + etiqueta + nivel de dificultad
        self.column_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'attack', 'level'
        ]
        
        # Columnas categóricas (copia local de la constante de módulo, para que
        # una mutación de la instancia no altere la fuente canónica)
        self.categorical_columns = list(COLUMNAS_CATEGORICAS)
        
        # Columnas numéricas (todas excepto las categóricas, attack y level)
        self.numerical_columns = [col for col in self.column_names 
                                if col not in self.categorical_columns + ['attack', 'level']]
        
        # Inicializar transformadores
        self.scaler = None
        self.label_encoder = None
        self.category_encoder = None
        self.feature_columns_after_encoding = None

        # Selección de características (4.3.5): lista final de features tras
        # el filtro varianza/correlación + importancias RF. None = sin aplicar.
        # feature_columns_after_encoding conserva SIEMPRE la lista completa
        # pre-selección (el scaler está ajustado sobre ella).
        self.selected_features = None
        
        # Mapeo de ataques a categorías principales
        self.attack_mapping = {
            'normal': 'normal',
            # DoS attacks
            'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
            'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos', 'apache2': 'dos',
            'processtable': 'dos', 'udpstorm': 'dos',
            # Probe attacks
            'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
            'mscan': 'probe', 'saint': 'probe',
            # R2L attacks
            'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l',
            'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l',
            'sendmail': 'r2l', 'named': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l',
            'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
            # U2R attacks
            'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
            'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
        }
        
    def load_dataset(self, train_path=os.path.join(RAIZ_REPO, "dataset", "KDDTrain+.txt"), test_path=os.path.join(RAIZ_REPO, "dataset", "KDDTest+.txt")):
        """
        Carga el dataset NSL-KDD desde archivos locales
        
        Parameters:
        -----------
        train_path : str
            Ruta al archivo de entrenamiento
        test_path : str
            Ruta al archivo de prueba
            
        Returns:
        --------
        tuple
            (train_df, test_df) DataFrames de pandas
        """
        try:
            # Detectar si son archivos .txt o .csv automáticamente
            if train_path.endswith('.txt'):
                # Para archivos .txt sin headers
                train_df = pd.read_csv(train_path, names=self.column_names, header=None)
                test_df = pd.read_csv(test_path, names=self.column_names, header=None)
            else:
                # Para archivos .csv (pueden tener headers)
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                
                # Si no tienen los nombres correctos, asignarlos
                if len(train_df.columns) == len(self.column_names):
                    train_df.columns = self.column_names
                    test_df.columns = self.column_names
            
            print(f"✓ Dataset cargado exitosamente!")
            print(f"  - Entrenamiento: {train_df.shape}")
            print(f"  - Prueba: {test_df.shape}")
            
            return train_df, test_df
            
        except Exception as e:
            print(f"❌ Error al cargar el dataset: {e}")
            return None, None

    def create_specialized_data_splits(self, train_df, test_df):
        """
        Crea las divisiones de datos especializadas D1, D2 y D3
        
        Parameters:
        -----------
        train_df : pandas.DataFrame
            DataFrame de entrenamiento
        test_df : pandas.DataFrame
            DataFrame de prueba
            
        Returns:
        --------
        dict
            Diccionario con las divisiones D1, D2 y D3
        """
        print("\n" + "="*80)
        print("CREANDO DIVISIONES DE DATOS ESPECIALIZADAS")
        print("="*80)
        
        # Agregar categorías de ataque a los DataFrames
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()
        
        if 'attack_category' not in train_df_copy.columns:
            train_df_copy['attack_category'] = train_df_copy['attack'].map(self.attack_mapping).fillna('unknown')
        if 'attack_category' not in test_df_copy.columns:
            test_df_copy['attack_category'] = test_df_copy['attack'].map(self.attack_mapping).fillna('unknown')
        
        # D1: Solo datos "Normal" del entrenamiento para modelo de anomalías
        print("\n📊 D1: Datos NORMAL para Entrenar Modelo de Anomalías")
        print("-" * 60)
        
        D1_normal = train_df_copy[train_df_copy['attack'] == 'normal'].copy()
        
        print(f"   ✓ Instancias normales seleccionadas: {len(D1_normal):,}")
        print(f"   ✓ Porcentaje del dataset de entrenamiento: {len(D1_normal)/len(train_df_copy)*100:.2f}%")
        
        # D2: Datos completos de prueba (Normal + Ataques) para evaluación general
        print("\n🎯 D2: Datos COMPLETOS de Prueba para Validación General")
        print("-" * 60)
        
        D2_complete_test = test_df_copy.copy()
        
        # Estadísticas de D2
        d2_normal_count = len(D2_complete_test[D2_complete_test['attack'] == 'normal'])
        d2_attack_count = len(D2_complete_test[D2_complete_test['attack'] != 'normal'])
        
        print(f"   ✓ Total de instancias: {len(D2_complete_test):,}")
        print(f"   ✓ Instancias normales: {d2_normal_count:,} ({d2_normal_count/len(D2_complete_test)*100:.2f}%)")
        print(f"   ✓ Instancias de ataque: {d2_attack_count:,} ({d2_attack_count/len(D2_complete_test)*100:.2f}%)")
        
        # Distribución por categorías en D2
        d2_category_dist = D2_complete_test['attack_category'].value_counts()
        print(f"   ✓ Distribución por categorías:")
        for category, count in d2_category_dist.items():
            print(f"      - {category}: {count:,} ({count/len(D2_complete_test)*100:.2f}%)")
        
        # D3: Datos de ataques conocidos del entrenamiento para modelo de firmas
        print("\n🔍 D3: Datos de ATAQUES CONOCIDOS para Entrenar Modelo de Firmas")
        print("-" * 60)
        
        D3_known_attacks = train_df_copy[train_df_copy['attack'] != 'normal'].copy()
        
        print(f"   ✓ Total de instancias de ataque: {len(D3_known_attacks):,}")
        print(f"   ✓ Porcentaje del dataset de entrenamiento: {len(D3_known_attacks)/len(train_df_copy)*100:.2f}%")
        
        # Distribución de ataques en D3
        d3_category_dist = D3_known_attacks['attack_category'].value_counts()
        d3_specific_dist = D3_known_attacks['attack'].value_counts()
        
        print(f"   ✓ Distribución por categorías principales:")
        for category, count in d3_category_dist.items():
            print(f"      - {category.upper()}: {count:,} ({count/len(D3_known_attacks)*100:.2f}%)")
        
        print(f"   ✓ Top 10 tipos específicos de ataques:")
        for attack, count in d3_specific_dist.head(10).items():
            category = self.attack_mapping.get(attack, 'unknown')
            print(f"      - {attack} ({category}): {count:,}")
        
        # Crear el diccionario de divisiones
        data_splits = {
            'D1_normal_for_anomaly': D1_normal,
            'D2_complete_test': D2_complete_test,
            'D3_known_attacks_for_signatures': D3_known_attacks
        }
        
        # Mostrar resumen final
        self._print_data_splits_summary(data_splits)
        
        return data_splits
    
    def _print_data_splits_summary(self, data_splits):
        """
        Imprime un resumen de las divisiones de datos creadas
        """
        print("\n" + "="*80)
        print("RESUMEN DE DIVISIONES DE DATOS CREADAS")
        print("="*80)
        
        print(f"\n📋 RESUMEN DE TAMAÑOS:")
        print(f"   D1 (Normal para Anomalías):     {len(data_splits['D1_normal_for_anomaly']):,} instancias")
        print(f"   D2 (Test Completo):             {len(data_splits['D2_complete_test']):,} instancias")
        print(f"   D3 (Ataques para Firmas):       {len(data_splits['D3_known_attacks_for_signatures']):,} instancias")
        
        print(f"\n🎯 PROPÓSITO DE CADA DIVISIÓN:")
        print(f"   D1: Entrenar modelo de detección de anomalías (solo comportamiento normal)")
        print(f"   D2: Evaluar rendimiento de ambos modelos (normal + ataques)")
        print(f"   D3: Extraer patrones/firmas para detección basada en reglas")
        
        print(f"\n📊 CATEGORÍAS DE ATAQUES EN D3:")
        d3_categories = data_splits['D3_known_attacks_for_signatures']['attack_category'].value_counts()
        for category, count in d3_categories.items():
            print(f"   - {category.upper()}: {count:,} instancias")
    
    def preprocess_specialized_splits(self, data_splits, scaler_type='standard'):
        """
        Preprocesa las divisiones especializadas de datos.

        El escalador se ajusta sobre D1+D3 (todos los datos de entrenamiento) para que
        los rangos de los ataques queden correctamente representados en la normalización.
        D2 (test) se transforma con ese mismo escalador sin re-ajustar.

        Parameters:
        -----------
        data_splits : dict
            Diccionario con las divisiones D1, D2 y D3
        scaler_type : str
            'minmax' → MinMaxScaler  |  'standard' → StandardScaler

        Returns:
        --------
        dict
            Diccionario con los datos preprocesados por división
        """
        print("\n" + "="*80)
        print("PREPROCESANDO DIVISIONES ESPECIALIZADAS")
        print("="*80)

        D1_KEY = 'D1_normal_for_anomaly'
        D2_KEY = 'D2_complete_test'
        D3_KEY = 'D3_known_attacks_for_signatures'

        # --- Paso 1: Separar features/labels y aplicar One-Hot Encoding ---
        print("\n🔧 Paso 1: Codificación de variables categóricas (One-Hot)...")
        encoded = {}
        for name, df in data_splits.items():
            df_copy = df.copy()
            drop_cols = [c for c in COLUMNAS_NO_CARACTERISTICA if c in df_copy.columns]
            X = df_copy.drop(drop_cols, axis=1)
            X_enc = pd.get_dummies(X, columns=self.categorical_columns, prefix=self.categorical_columns)
            encoded[name] = {
                'X_enc': X_enc,
                'y_attack': df_copy['attack'],
                'y_category': df_copy['attack_category'],
                'index': df_copy.index,
            }
            print(f"   📊 {name}: {X_enc.shape[0]:,} instancias, {X_enc.shape[1]} características")

        # --- Paso 2: Alinear columnas usando la unión del train (D1 + D3) ---
        # El vocabulario de columnas tras el one-hot es la UNIÓN de D1 y D3 (todo el
        # entrenamiento), con orden determinista (sorted). Las categorías exclusivas
        # de D2 (test) se descartan a propósito: en producción tampoco se conocerían.
        # No es leakage, es realismo.
        print("\n🔧 Paso 2: Alineando columnas (unión del train D1 + D3)...")
        train_cols = sorted(set(encoded[D1_KEY]['X_enc'].columns) | set(encoded[D3_KEY]['X_enc'].columns))
        self.feature_columns_after_encoding = train_cols
        for name, data in encoded.items():
            for col in self.feature_columns_after_encoding:
                if col not in data['X_enc'].columns:
                    data['X_enc'][col] = 0
            data['X_enc'] = data['X_enc'][self.feature_columns_after_encoding]
        print(f"   ✓ {len(self.feature_columns_after_encoding)} características finales")

        # --- Paso 3: Ajustar escalador sobre D1+D3 (todo el entrenamiento) ---
        print("\n🔧 Paso 3: Ajustando escalador sobre D1 + D3...")
        X_train_all = pd.concat(
            [encoded[D1_KEY]['X_enc'], encoded[D3_KEY]['X_enc']], axis=0
        )
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        self.scaler.fit(X_train_all)
        print(f"   ✓ {type(self.scaler).__name__} ajustado sobre {len(X_train_all):,} instancias (D1+D3)")

        # --- Paso 4: Ajustar codificadores de etiquetas ---
        print("\n🔧 Paso 4: Ajustando codificadores de etiquetas...")
        all_attacks = sorted(self.attack_mapping.keys())
        all_categories = sorted(set(self.attack_mapping.values()))

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_attacks)

        self.category_encoder = LabelEncoder()
        self.category_encoder.fit(all_categories)

        print(f"   ✓ LabelEncoder: {len(all_attacks)} tipos de ataque")
        print(f"   ✓ CategoryEncoder: {len(all_categories)} categorías")

        # --- Paso 5: Transformar todas las divisiones ---
        print("\n🔧 Paso 5: Transformando D1, D2 y D3...")
        processed_splits = {}
        for name, data in encoded.items():
            X_scaled = self.scaler.transform(data['X_enc'])
            X_scaled_df = pd.DataFrame(
                X_scaled,
                columns=self.feature_columns_after_encoding,
                index=data['index'],
            )
            y_attack_enc = self.label_encoder.transform(data['y_attack'])
            y_category_enc = self.category_encoder.transform(data['y_category'])

            processed_splits[name] = {
                'X': X_scaled_df,
                'y_attack': y_attack_enc,
                'y_category': y_category_enc,
                'y_attack_original': data['y_attack'].values,
                'y_category_original': data['y_category'].values,
                'original_indices': data['index'].values,
            }
            print(f"   ✓ {name}: {X_scaled_df.shape[0]:,} × {X_scaled_df.shape[1]}")

        self._print_preprocessing_summary(processed_splits)
        return processed_splits
    
    def _print_preprocessing_summary(self, processed_splits):
        """
        Imprime resumen del preprocesamiento de las divisiones
        """
        print("\n" + "="*80)
        print("RESUMEN DEL PREPROCESAMIENTO DE DIVISIONES")
        print("="*80)
        
        print(f"\n📊 DIMENSIONES DESPUÉS DEL PREPROCESAMIENTO:")
        for split_name, split_data in processed_splits.items():
            X_shape = split_data['X'].shape
            print(f"   {split_name}: {X_shape[0]:,} × {X_shape[1]} características")
        
        print(f"\n🔧 TRANSFORMACIONES APLICADAS:")
        print(f"   - Escalador: {type(self.scaler).__name__}")
        print(f"   - Características finales: {len(self.feature_columns_after_encoding)}")
        print(f"   - Clases de ataque: {len(self.label_encoder.classes_)}")
        print(f"   - Categorías de ataque: {len(self.category_encoder.classes_)}")
        
        print(f"\n📋 DISTRIBUCIONES POR DIVISIÓN:")
        for split_name, split_data in processed_splits.items():
            print(f"\n   {split_name}:")
            unique_categories, counts = np.unique(split_data['y_category_original'], return_counts=True)
            for cat, count in zip(unique_categories, counts):
                percentage = count / len(split_data['y_category_original']) * 100
                print(f"      - {cat}: {count:,} ({percentage:.2f}%)")
    
    def select_features(self, processed_splits, var_threshold=1e-8, corr_threshold=0.95,
                        importancia_acumulada=0.999,
                        output_dir=os.path.join(RAIZ_REPO, "Resultados")):
        """
        Selección de características (sección 4.3.5 de la memoria).

        Se ejecuta DESPUÉS del escalado y consta de dos pasos:

        1) Filtro barato, calculado sobre D1+D3 (train completo):
           - se eliminan las características con varianza ≈ 0
           - de cada par con |correlación| > corr_threshold se conserva la de
             mayor varianza
           ⚠️ NUNCA calcular este filtro solo sobre D1: los dummies de
           service/flag exclusivos de tráfico de ataque son todo-cero en D1
           (solo normal) y el filtro los eliminaría, deshaciendo silenciosamente
           el vocabulario one-hot construido sobre la unión de D1 y D3.

        2) Importancias de un RandomForestClassifier entrenado sobre D1+D3 con
           y_category (5 clases: normal + 4 categorías de ataque): se conserva
           el top-N de características que acumula ~importancia_acumulada de la
           importancia total.

           El umbral por defecto es el 99.9% (no el 99%): el corte al 99%
           eliminaba indicadores de ataques no vistos (num_failed_logins,
           flag_REJ, flag_SH, num_shells...) que son ~0 en D1 y que el RF
           supervisado infravalora: u2r tiene solo 52 muestras —cifra exacta,
           ver Resultados/specialized_nsl_kdd_composicion_d3.csv— y el criterio
           de impureza está sesgado contra las dummies binarias raras. El
           umbral se contrasta con el experimento de 54 frente a 122 features,
           que mide el recall 0-day por tipo de ataque.

        Modifica processed_splits in-place: las X de D1/D2/D3 quedan filtradas a
        las características seleccionadas (conservando el orden original de
        columnas), de modo que los scripts posteriores (anomalias.py, firmas.py)
        no necesitan saber nada de la selección.

        Escribe <output_dir>/selected_features.txt con las conservadas (y su
        importancia) y las eliminadas con el motivo de cada una.

        Parameters:
        -----------
        processed_splits : dict
            Divisiones ya escaladas devueltas por preprocess_specialized_splits
        var_threshold : float
            Umbral por debajo del cual la varianza se considera ~0
        corr_threshold : float
            Umbral de |correlación| para el filtro de pares redundantes
        importancia_acumulada : float
            Fracción de la importancia RF total que debe acumular el top-N
            (por defecto 0.999 → 99.9%; ver justificación arriba)
        output_dir : str
            Directorio donde se escribe selected_features.txt

        Returns:
        --------
        list
            Lista de características seleccionadas (en el orden original)
        """
        print("\n" + "="*80)
        print("SELECCIÓN DE CARACTERÍSTICAS (4.3.5)")
        print("="*80)

        D1_KEY = 'D1_normal_for_anomaly'
        D3_KEY = 'D3_known_attacks_for_signatures'

        features_iniciales = list(self.feature_columns_after_encoding)
        n_inicial = len(features_iniciales)

        # Base de cálculo: train completo (D1+D3). Ver advertencia del docstring.
        X_train = pd.concat(
            [processed_splits[D1_KEY]['X'], processed_splits[D3_KEY]['X']], axis=0
        )
        y_train = np.concatenate([
            processed_splits[D1_KEY]['y_category_original'],
            processed_splits[D3_KEY]['y_category_original'],
        ])

        # --- Paso 1a: filtro de varianza ~0 (sobre D1+D3) ---
        print(f"\n🔧 Paso 1a: Filtro de varianza ~0 (calculada sobre D1+D3)...")
        varianzas = X_train.var()
        eliminadas_varianza = [f for f in features_iniciales if varianzas[f] < var_threshold]
        restantes = [f for f in features_iniciales if f not in set(eliminadas_varianza)]
        print(f"   ✓ Eliminadas {len(eliminadas_varianza)} características con varianza < {var_threshold}")

        # --- Paso 1b: filtro de correlación (sobre D1+D3) ---
        # De cada par con |corr| > umbral se elimina la de MENOR varianza.
        print(f"\n🔧 Paso 1b: Filtro de correlación |corr| > {corr_threshold} (sobre D1+D3)...")
        corr = X_train[restantes].corr().abs()
        eliminadas_corr = {}  # feature eliminada -> (feature con la que correlaciona, |corr|)
        activas = set(restantes)
        cols = list(corr.columns)
        for i in range(len(cols)):
            if cols[i] not in activas:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] not in activas:
                    continue
                if corr.iloc[i, j] > corr_threshold:
                    # Conservar la de mayor varianza del par
                    if varianzas[cols[i]] >= varianzas[cols[j]]:
                        activas.discard(cols[j])
                        eliminadas_corr[cols[j]] = (cols[i], float(corr.iloc[i, j]))
                    else:
                        activas.discard(cols[i])
                        eliminadas_corr[cols[i]] = (cols[j], float(corr.iloc[i, j]))
                        break  # cols[i] eliminada: pasar a la siguiente
        restantes = [f for f in restantes if f in activas]
        print(f"   ✓ Eliminadas {len(eliminadas_corr)} características redundantes por correlación")

        # --- Paso 2: importancias de RandomForest (sobre D1+D3, y_category) ---
        print(f"\n🔧 Paso 2: Importancias RF (top-N que acumule ~{importancia_acumulada*100:g}%)...")
        rf = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
        )
        rf.fit(X_train[restantes], y_train)
        importancias = pd.Series(rf.feature_importances_, index=restantes).sort_values(ascending=False)
        acumulada = importancias.cumsum()

        # Conservar el mínimo top-N cuya importancia acumulada alcanza el umbral
        n_top = int((acumulada < importancia_acumulada).sum()) + 1
        n_top = min(n_top, len(importancias))
        seleccionadas_set = set(importancias.index[:n_top])
        eliminadas_importancia = [f for f in restantes if f not in seleccionadas_set]
        print(f"   ✓ RF entrenado sobre {len(X_train):,} instancias (D1+D3), {len(set(y_train))} clases")
        print(f"   ✓ Top-{n_top} características acumulan {acumulada.iloc[n_top-1]*100:.2f}% de la importancia")

        # Lista final conservando el orden original de columnas (determinista)
        self.selected_features = [f for f in features_iniciales if f in seleccionadas_set]

        # --- Aplicar la selección a las tres divisiones (in-place) ---
        for name in processed_splits:
            processed_splits[name]['X'] = processed_splits[name]['X'][self.selected_features]

        # --- Persistir el detalle en selected_features.txt ---
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'selected_features.txt')
        self._save_feature_selection_report(
            report_path, n_inicial, importancias, acumulada, n_top,
            eliminadas_varianza, eliminadas_corr, eliminadas_importancia,
            varianzas, var_threshold, corr_threshold, importancia_acumulada
        )

        # --- Resumen por consola ---
        print(f"\n📊 RESUMEN DE LA SELECCIÓN DE CARACTERÍSTICAS:")
        print(f"   Características de partida:        {n_inicial}")
        print(f"   Eliminadas por varianza ~0:        {len(eliminadas_varianza)}")
        print(f"   Eliminadas por correlación >{corr_threshold}: {len(eliminadas_corr)}")
        print(f"   Eliminadas por importancia (fuera del top-{importancia_acumulada*100:g}%): {len(eliminadas_importancia)}")
        print(f"   Características finales:           {len(self.selected_features)}")
        print(f"   📄 Detalle guardado en: {report_path}")

        return self.selected_features

    @staticmethod
    def _resolver_representante_conservado(feat, eliminadas_corr):
        """
        Sigue la cadena de eliminaciones greedy del filtro de correlación
        (A eliminada por B, B eliminada a su vez por C, ...) hasta la
        característica que realmente SOBREVIVIÓ al filtro, para que el reporte
        no afirme que la pareja "se conservó" cuando también fue eliminada más
        adelante en el barrido.

        La protección contra ciclos es defensiva: el barrido greedy no debería
        producirlos, pero si apareciera uno se devuelve la última visitada.
        """
        visitadas = {feat}
        actual = eliminadas_corr[feat][0]
        while actual in eliminadas_corr and actual not in visitadas:
            visitadas.add(actual)
            actual = eliminadas_corr[actual][0]
        return actual

    def _save_feature_selection_report(self, report_path, n_inicial, importancias, acumulada,
                                       n_top, eliminadas_varianza, eliminadas_corr,
                                       eliminadas_importancia, varianzas, var_threshold,
                                       corr_threshold, importancia_acumulada):
        """
        Escribe selected_features.txt: características conservadas con su
        importancia RF, y eliminadas con el motivo (varianza ~0 / correlación /
        fuera del top de importancia acumulada).
        """
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SELECCIÓN DE CARACTERÍSTICAS — 4.3.5 (NSL-KDD)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Fecha de creación: {pd.Timestamp.now()}\n")
            f.write("Base de cálculo: D1+D3 (train completo) tras el escalado.\n")
            f.write("Ranking: RandomForestClassifier(n_estimators=100, random_state=42, "
                    "class_weight='balanced') con y_category (5 clases).\n\n")

            f.write(f"Características de partida:  {n_inicial}\n")
            f.write(f"Eliminadas por varianza ~0 (< {var_threshold}): {len(eliminadas_varianza)}\n")
            f.write(f"Eliminadas por correlación (> {corr_threshold}): {len(eliminadas_corr)}\n")
            f.write(f"Eliminadas por importancia (fuera del top-{importancia_acumulada*100:g}%): "
                    f"{len(eliminadas_importancia)}\n")
            f.write(f"Características finales:     {len(self.selected_features)}\n\n")

            f.write(f"1. CARACTERÍSTICAS CONSERVADAS ({len(self.selected_features)}) — "
                    f"ordenadas por importancia RF:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'#':>4}  {'importancia':>12}  {'acumulada':>10}  feature\n")
            for rank, feat in enumerate(importancias.index[:n_top], start=1):
                f.write(f"{rank:>4}  {importancias[feat]:>12.6f}  {acumulada[feat]:>10.4f}  {feat}\n")

            f.write(f"\n2. ELIMINADAS POR VARIANZA ~0 sobre D1+D3 ({len(eliminadas_varianza)}):\n")
            f.write("-" * 80 + "\n")
            if eliminadas_varianza:
                for feat in eliminadas_varianza:
                    f.write(f"   - {feat} (varianza={varianzas[feat]:.2e})\n")
            else:
                f.write("   (ninguna)\n")

            f.write(f"\n3. ELIMINADAS POR CORRELACIÓN > {corr_threshold} ({len(eliminadas_corr)}):\n")
            f.write("-" * 80 + "\n")
            f.write("   De cada par se conservó la característica de mayor varianza.\n")
            f.write("   Si la pareja de referencia fue eliminada a su vez más adelante en el\n")
            f.write("   barrido greedy (cadena A→B→C...), se indica el representante que\n")
            f.write("   realmente sobrevivió al filtro de correlación.\n")
            if eliminadas_corr:
                descartadas_rf = set(eliminadas_importancia)
                for feat, (pareja, valor) in sorted(eliminadas_corr.items()):
                    if pareja in eliminadas_corr:
                        # La pareja también acabó eliminada: resolver la cadena
                        # hasta el representante que sí superó este filtro
                        representante = self._resolver_representante_conservado(feat, eliminadas_corr)
                        detalle = (f"con '{pareja}' (eliminada a su vez por correlación; "
                                   f"representante final que superó este filtro: '{representante}')")
                    else:
                        representante = pareja
                        detalle = f"con '{pareja}', que superó este filtro"
                    if representante in descartadas_rf:
                        # Veracidad completa: superar el filtro de correlación no
                        # implica llegar a la lista final si el RF la descartó después
                        detalle += (f" [nota: '{representante}' fue descartada después "
                                    f"por el filtro de importancia RF]")
                    f.write(f"   - {feat} (|corr|={valor:.4f} {detalle})\n")
            else:
                f.write("   (ninguna)\n")

            f.write(f"\n4. ELIMINADAS POR IMPORTANCIA — fuera del top que acumula "
                    f"~{importancia_acumulada*100:g}% ({len(eliminadas_importancia)}):\n")
            f.write("-" * 80 + "\n")
            if eliminadas_importancia:
                for feat in sorted(eliminadas_importancia, key=lambda x: -importancias[x]):
                    f.write(f"   - {feat} (importancia={importancias[feat]:.6f})\n")
            else:
                f.write("   (ninguna)\n")

    def save_specialized_splits(self, original_splits, processed_splits, base_path='specialized_nsl_kdd'):
        """
        Guarda todas las divisiones especializadas (originales y procesadas)
        """
        print("\n" + "="*80)
        print("GUARDANDO DIVISIONES ESPECIALIZADAS")
        print("="*80)
        
        output_directory = os.path.dirname(base_path)
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"   ✓ Directorio creado: {output_directory}")
        
        saved_files = []
        
        # Guardar datos originales
        print("\n💾 Guardando datos originales...")
        for split_name, split_df in original_splits.items():
            filename = f"{base_path}_original_{split_name}.csv"
            split_df.to_csv(filename, index=False)
            saved_files.append(filename)
            print(f"   ✓ {split_name}: {split_df.shape}")
        
        # Guardar datos procesados
        print("\n💾 Guardando datos procesados...")
        for split_name, split_data in processed_splits.items():
            # Características (X)
            x_filename = f"{base_path}_processed_X_{split_name}.csv"
            split_data['X'].to_csv(x_filename, index=False)
            saved_files.append(x_filename)
            
            # Etiquetas de ataque específico
            y_attack_filename = f"{base_path}_processed_y_attack_{split_name}.csv"
            pd.DataFrame({
                'attack_encoded': split_data['y_attack'],
                'attack_original': split_data['y_attack_original']
            }).to_csv(y_attack_filename, index=False)
            saved_files.append(y_attack_filename)
            
            # Etiquetas de categoría
            y_category_filename = f"{base_path}_processed_y_category_{split_name}.csv"
            pd.DataFrame({
                'category_encoded': split_data['y_category'],
                'category_original': split_data['y_category_original']
            }).to_csv(y_category_filename, index=False)
            saved_files.append(y_category_filename)
            
            print(f"   ✓ {split_name}: X{split_data['X'].shape}, y_attack({len(split_data['y_attack'])}), y_category({len(split_data['y_category'])})")
        
        # Guardar información de mapeos y transformaciones
        print("\n💾 Guardando información de transformaciones...")
        self._save_specialized_mappings(base_path)

        # Persistir los transformadores ajustados (scaler, encoders y vocabulario
        # de columnas) para que los scripts de modelos los reutilicen sin re-ajustar.
        # - 'feature_columns': lista FINAL (tras selección 4.3.5 si se aplicó) —
        #   es la que consumen anomalias.py / firmas.py y la que validan los CSVs.
        # - 'feature_columns_pre_seleccion': lista completa post one-hot — el
        #   scaler está ajustado sobre ella y hace falta para transformar datos
        #   nuevos antes de filtrar a las seleccionadas.
        columnas_finales = (self.selected_features
                            if self.selected_features is not None
                            else self.feature_columns_after_encoding)
        transformers_file = f'{base_path}_transformers.joblib'
        joblib.dump({
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'category_encoder': self.category_encoder,
            'feature_columns': columnas_finales,
            'feature_columns_pre_seleccion': self.feature_columns_after_encoding
        }, transformers_file)
        saved_files.append(transformers_file)
        print(f"   ✓ Transformadores persistidos: {transformers_file}")

        # Guardar resumen de uso
        self._save_usage_guide(base_path, saved_files)
        
        print(f"\n✅ TODAS LAS DIVISIONES GUARDADAS EXITOSAMENTE")
        print(f"   📁 Archivos base: {base_path}_*")
        print(f"   📊 Total de archivos creados: {len(saved_files) + 2}")  # +2 por mappings y guía de uso
        
        return saved_files
    
    def _save_specialized_mappings(self, base_path):
        """
        Guarda los mapeos específicos para las divisiones especializadas
        """
        mappings_file = f'{base_path}_mappings_and_info.txt'
        
        with open(mappings_file, 'w', encoding='utf-8') as f:
            f.write("MAPEOS Y TRANSFORMACIONES - DIVISIONES ESPECIALIZADAS NSL-KDD\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Fecha de creación: {pd.Timestamp.now()}\n\n")
            
            # Mapeo de ataques específicos a códigos
            f.write("1. MAPEO DE ATAQUES ESPECÍFICOS (LABEL ENCODING):\n")
            f.write("-" * 50 + "\n")
            for i, attack in enumerate(self.label_encoder.classes_):
                f.write(f"   {i:2d} → {attack}\n")
            f.write(f"\nTotal de ataques específicos: {len(self.label_encoder.classes_)}\n\n")
            
            # Mapeo de categorías de ataques
            f.write("2. MAPEO DE CATEGORÍAS DE ATAQUES:\n")
            f.write("-" * 40 + "\n")
            for i, category in enumerate(self.category_encoder.classes_):
                f.write(f"   {i} → {category.upper()}\n")
            f.write(f"\nTotal de categorías: {len(self.category_encoder.classes_)}\n\n")
            
            # Mapeo de ataques específicos a categorías
            f.write("3. MAPEO ATAQUE ESPECÍFICO → CATEGORÍA:\n")
            f.write("-" * 45 + "\n")
            current_category = None
            for attack, category in sorted(self.attack_mapping.items(), key=lambda x: x[1]):
                if category != current_category:
                    f.write(f"\n   {category.upper()}:\n")
                    current_category = category
                f.write(f"      - {attack}\n")
            
            # Información de escalador
            f.write(f"\n4. INFORMACIÓN DEL ESCALADOR:\n")
            f.write("-" * 30 + "\n")
            f.write(f"   Tipo: {type(self.scaler).__name__}\n")
            if hasattr(self.scaler, 'feature_range'):
                f.write(f"   Rango: {self.scaler.feature_range}\n")
            f.write(f"   Características tras one-hot: {len(self.feature_columns_after_encoding)}\n")
            if self.selected_features is not None:
                f.write(f"   Características tras selección (4.3.5): {len(self.selected_features)}\n")
                f.write("   Detalle de la selección: selected_features.txt\n")
            else:
                f.write("   Selección de características (4.3.5): NO aplicada en esta ejecución\n")
            f.write("\n")
            
            # División de datos
            f.write("5. DESCRIPCIÓN DE DIVISIONES:\n")
            f.write("-" * 30 + "\n")
            f.write("   D1 (normal_for_anomaly):\n")
            f.write("      - Solo instancias 'normal' del entrenamiento\n")
            f.write("      - Para entrenar modelos de detección de anomalías\n")
            f.write("      - El modelo aprende qué es comportamiento legítimo\n\n")
            
            f.write("   D2 (complete_test):\n")
            f.write("      - Dataset completo de prueba (normal + ataques)\n")
            f.write("      - Para evaluación general de ambos modelos\n")
            f.write("      - Permite medir rendimiento real\n\n")
            
            f.write("   D3 (known_attacks_for_signatures):\n")
            f.write("      - Solo ataques del dataset de entrenamiento\n")
            f.write("      - Para extraer patrones/firmas de ataques conocidos\n")
            f.write("      - Base para sistema de detección basado en reglas\n")
    
    def _save_usage_guide(self, base_path, saved_files):
        """
        Guarda una guía de uso de los archivos generados
        """
        guide_file = f'{base_path}_usage_guide.txt'
        
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write("GUÍA DE USO - DIVISIONES ESPECIALIZADAS NSL-KDD\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("📋 ARCHIVOS GENERADOS:\n")
            f.write("-" * 20 + "\n")
            f.write("DATOS ORIGINALES:\n")
            f.write("   - *_original_D1_normal_for_anomaly.csv\n")
            f.write("   - *_original_D2_complete_test.csv\n")
            f.write("   - *_original_D3_known_attacks_for_signatures.csv\n\n")
            
            f.write("DATOS PROCESADOS:\n")
            f.write("   - *_processed_X_[división].csv (características)\n")
            f.write("   - *_processed_y_attack_[división].csv (etiquetas específicas)\n")
            f.write("   - *_processed_y_category_[división].csv (categorías)\n\n")
            
            f.write("INFORMACIÓN Y MAPEOS:\n")
            f.write("   - *_mappings_and_info.txt (este archivo)\n")
            f.write("   - *_usage_guide.txt (guía de uso)\n\n")
            
            f.write("🎯 CÓMO USAR CADA DIVISIÓN:\n")
            f.write("-" * 30 + "\n")
            f.write("1. MODELO DE ANOMALÍAS (D1):\n")
            f.write("   import pandas as pd\n")
            f.write(f"   X_normal = pd.read_csv('{base_path}_processed_X_D1_normal_for_anomaly.csv')\n")
            f.write("   # Entrenar modelo solo con datos normales\n")
            f.write("   # Ejemplo: IsolationForest, OneClassSVM, Autoencoder\n\n")
            
            f.write("2. EVALUACIÓN GENERAL (D2):\n")
            f.write(f"   X_test = pd.read_csv('{base_path}_processed_X_D2_complete_test.csv')\n")
            f.write(f"   y_test = pd.read_csv('{base_path}_processed_y_category_D2_complete_test.csv')\n")
            f.write("   # Evaluar ambos modelos (anomalías + firmas)\n")
            f.write("   # Métricas: precisión, recall, F1-score por categoría\n\n")
            
            f.write("3. EXTRACCIÓN DE FIRMAS (D3):\n")
            f.write(f"   X_attacks = pd.read_csv('{base_path}_processed_X_D3_known_attacks_for_signatures.csv')\n")
            f.write(f"   y_attacks = pd.read_csv('{base_path}_processed_y_category_D3_known_attacks_for_signatures.csv')\n")
            f.write("   # Analizar patrones por categoría de ataque\n")
            f.write("   # Crear reglas/firmas basadas en características distintivas\n\n")
            
            f.write("💡 EJEMPLO DE IMPLEMENTACIÓN COMPLETA:\n")
            f.write("-" * 40 + "\n")
            f.write("```python\n")
            f.write("# 1. Cargar divisiones\n")
            f.write("import pandas as pd\n")
            f.write("from sklearn.ensemble import IsolationForest\n")
            f.write("from sklearn.metrics import classification_report\n\n")
            
            f.write("# Datos para modelo de anomalías\n")
            f.write(f"X_normal = pd.read_csv('{base_path}_processed_X_D1_normal_for_anomaly.csv')\n\n")
            
            f.write("# Datos para evaluación\n")
            f.write(f"X_test = pd.read_csv('{base_path}_processed_X_D2_complete_test.csv')\n")
            f.write(f"y_test = pd.read_csv('{base_path}_processed_y_category_D2_complete_test.csv')\n\n")
            
            f.write("# Datos para firmas\n")
            f.write(f"X_attacks = pd.read_csv('{base_path}_processed_X_D3_known_attacks_for_signatures.csv')\n")
            f.write(f"y_attacks = pd.read_csv('{base_path}_processed_y_category_D3_known_attacks_for_signatures.csv')\n\n")
            
            f.write("# 2. Entrenar modelo de anomalías\n")
            f.write("anomaly_model = IsolationForest(contamination=0.1, random_state=42)\n")
            f.write("anomaly_model.fit(X_normal)  # Solo con datos normales\n\n")
            
            f.write("# 3. Extraer firmas de ataques\n")
            f.write("# Análisis estadístico por categoría\n")
            f.write("for category in y_attacks['category_original'].unique():\n")
            f.write("    mask = y_attacks['category_original'] == category\n")
            f.write("    category_data = X_attacks[mask]\n")
            f.write("    # Análizar características distintivas\n")
            f.write("    print(f'Patrones para {category}: {category_data.describe()}')\n\n")
            
            f.write("# 4. Evaluar en datos de prueba\n")
            f.write("anomaly_predictions = anomaly_model.predict(X_test)\n")
            f.write("# -1 = anomalía, 1 = normal\n")
            f.write("print('Evaluación del modelo de anomalías:')\n")
            f.write("# Convertir a binario para evaluación\n")
            f.write("y_binary = (y_test['category_original'] != 'normal').astype(int)\n")
            f.write("anomaly_binary = (anomaly_predictions == -1).astype(int)\n")
            f.write("print(classification_report(y_binary, anomaly_binary))\n")
            f.write("```\n\n")
            
            f.write("🔧 CONSIDERACIONES TÉCNICAS:\n")
            f.write("-" * 30 + "\n")
            f.write("- D1 está balanceado (solo 'normal') para entrenar detección de anomalías\n")
            f.write("- D2 mantiene la distribución original para evaluación realista\n")
            f.write("- D3 permite análisis detallado de patrones de ataque\n")
            f.write("- Los encoders están ajustados consistentemente en todas las divisiones\n")
            f.write("- Las características están escaladas uniformemente\n\n")
            
            f.write("⚠️  IMPORTANTES:\n")
            f.write("-" * 15 + "\n")
            f.write("- Siempre usar D1 SOLO para entrenar modelos de anomalías\n")
            f.write("- D2 es para evaluación final de ambos sistemas\n")
            f.write("- D3 permite crear reglas complementarias al modelo de anomalías\n")
            f.write("- Los índices originales se mantienen para trazabilidad\n")

    def exploratory_data_analysis(self, train_df, test_df):
        """
        Realiza análisis exploratorio de datos (EDA) con enfoque en las divisiones especializadas
        """
        print("="*80)
        print("ANÁLISIS EXPLORATORIO DE DATOS (EDA) - ENFOQUE ESPECIALIZADO")
        print("="*80)
        
        # EDA básico (versión simplificada del original)
        print("\n1. DIMENSIONES DE LOS DATASETS:")
        print(f"   Entrenamiento: {train_df.shape[0]:,} filas × {train_df.shape[1]} columnas")
        print(f"   Prueba: {test_df.shape[0]:,} filas × {test_df.shape[1]} columnas")
        
        # Agregar categorías de ataque
        train_df_analysis = train_df.copy()
        test_df_analysis = test_df.copy()
        
        train_df_analysis['attack_category'] = train_df_analysis['attack'].map(self.attack_mapping).fillna('unknown')
        test_df_analysis['attack_category'] = test_df_analysis['attack'].map(self.attack_mapping).fillna('unknown')
        
        # Análisis específico para las divisiones
        print("\n2. ANÁLISIS PARA DIVISIONES ESPECIALIZADAS:")
        print("-" * 50)
        
        # Análisis D1 (Normal)
        normal_data = train_df_analysis[train_df_analysis['attack'] == 'normal']
        print(f"\n   D1 - Datos Normal para Anomalías:")
        print(f"   ✓ Instancias disponibles: {len(normal_data):,}")
        print(f"   ✓ Porcentaje del total de entrenamiento: {len(normal_data)/len(train_df_analysis)*100:.2f}%")
        
        # Análisis D2 (Test completo)  
        print(f"\n   D2 - Test Completo para Evaluación:")
        print(f"   ✓ Total de instancias: {len(test_df_analysis):,}")
        test_normal = len(test_df_analysis[test_df_analysis['attack'] == 'normal'])
        test_attacks = len(test_df_analysis) - test_normal
        print(f"   ✓ Normal: {test_normal:,} ({test_normal/len(test_df_analysis)*100:.2f}%)")
        print(f"   ✓ Ataques: {test_attacks:,} ({test_attacks/len(test_df_analysis)*100:.2f}%)")
        
        # Análisis D3 (Ataques conocidos)
        attack_data = train_df_analysis[train_df_analysis['attack'] != 'normal']
        print(f"\n   D3 - Ataques Conocidos para Firmas:")
        print(f"   ✓ Total de ataques: {len(attack_data):,}")
        print(f"   ✓ Porcentaje del entrenamiento: {len(attack_data)/len(train_df_analysis)*100:.2f}%")
        
        # Distribución por categorías en D3
        d3_categories = attack_data['attack_category'].value_counts()
        print(f"   ✓ Distribución por categorías:")
        for category, count in d3_categories.items():
            print(f"      - {category.upper()}: {count:,} ({count/len(attack_data)*100:.2f}%)")
        
        # Visualizaciones específicas para las divisiones
        self._plot_specialized_distributions(train_df_analysis, test_df_analysis)
        
        print("\n3. VERIFICACIONES DE CONSISTENCIA:")
        print("-" * 40)
        
        # Verificar ataques únicos
        train_attacks = set(train_df_analysis['attack'].unique())
        test_attacks = set(test_df_analysis['attack'].unique())
        
        only_train = train_attacks - test_attacks
        only_test = test_attacks - train_attacks
        common = train_attacks & test_attacks
        
        print(f"   ✓ Ataques solo en entrenamiento: {len(only_train)}")
        print(f"   ✓ Ataques solo en prueba: {len(only_test)}")
        print(f"   ✓ Ataques comunes: {len(common)}")
        
        if only_test:
            print(f"   ⚠️  Ataques nuevos en prueba: {sorted(list(only_test))}")
            print(f"      → Estos requerirán detección por anomalías")
        
        return train_df_analysis, test_df_analysis
    
    def _plot_specialized_distributions(self, train_df, test_df):
        """
        Crea visualizaciones específicas para las divisiones especializadas
        """
        plt.figure(figsize=(20, 15))
        
        # Subplot 1: Distribución para D1 (solo mostrar que es 100% normal)
        plt.subplot(3, 3, 1)
        plt.bar(['Normal'], [len(train_df[train_df['attack'] == 'normal'])], color='lightgreen', alpha=0.7)
        plt.title('D1: Datos para Modelo de Anomalías\n(Solo Normal)')
        plt.ylabel('Frecuencia')
        
        # Subplot 2: Distribución D2 (test completo)
        plt.subplot(3, 3, 2)
        test_category_counts = test_df['attack_category'].value_counts()
        colors = ['lightgreen' if cat == 'normal' else 'lightcoral' for cat in test_category_counts.index]
        plt.bar(test_category_counts.index, test_category_counts.values, color=colors, alpha=0.7)
        plt.title('D2: Test Completo para Evaluación\n(Normal + Ataques)')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=45)
        
        # Subplot 3: Distribución D3 (ataques para firmas)
        plt.subplot(3, 3, 3)
        attack_data = train_df[train_df['attack'] != 'normal']
        d3_category_counts = attack_data['attack_category'].value_counts()
        plt.bar(d3_category_counts.index, d3_category_counts.values, color='lightcoral', alpha=0.7)
        plt.title('D3: Ataques para Firmas\n(Solo Ataques Conocidos)')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=45)
        
        # Subplot 4: Comparación de tamaños
        plt.subplot(3, 3, 4)
        d1_size = len(train_df[train_df['attack'] == 'normal'])
        d2_size = len(test_df)
        d3_size = len(attack_data)
        
        plt.bar(['D1\n(Normal)', 'D2\n(Test)', 'D3\n(Ataques)'], 
                [d1_size, d2_size, d3_size], 
                color=['lightgreen', 'lightblue', 'lightcoral'], alpha=0.7)
        plt.title('Comparación de Tamaños\nde Divisiones')
        plt.ylabel('Número de Instancias')
        plt.yscale('log')
        
        # Subplot 5: Distribución de ataques específicos en D3
        plt.subplot(3, 3, 5)
        top_attacks_d3 = attack_data['attack'].value_counts().head(8)
        plt.barh(range(len(top_attacks_d3)), top_attacks_d3.values)
        plt.yticks(range(len(top_attacks_d3)), top_attacks_d3.index)
        plt.title('Top 8 Ataques Específicos en D3')
        plt.xlabel('Frecuencia')
        plt.gca().invert_yaxis()
        
        # Subplot 6: Distribución porcentual por categorías
        plt.subplot(3, 3, 6)
        # Calcular porcentajes para cada división
        d1_pct = [100]  # Solo normal
        d2_category_pct = (test_df['attack_category'].value_counts() / len(test_df) * 100).values
        d3_category_pct = (d3_category_counts / len(attack_data) * 100).values
        
        categories = ['normal', 'dos', 'probe', 'r2l', 'u2r']
        d2_pcts = [test_df['attack_category'].value_counts().get(cat, 0) / len(test_df) * 100 for cat in categories]
        d3_pcts = [d3_category_counts.get(cat, 0) / len(attack_data) * 100 for cat in categories if cat != 'normal']
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, d2_pcts, width, label='D2 (Test)', alpha=0.8)
        d3_pcts_with_zero = [0] + d3_pcts  # Agregar 0 para 'normal' en D3
        plt.bar(x + width/2, d3_pcts_with_zero, width, label='D3 (Ataques)', alpha=0.8)
        
        plt.xlabel('Categorías')
        plt.ylabel('Porcentaje (%)')
        plt.title('Distribución Porcentual\nD2 vs D3')
        plt.xticks(x, categories)
        plt.legend()
        
        # Subplot 7: Características numéricas importantes en D1
        plt.subplot(3, 3, 7)
        normal_data = train_df[train_df['attack'] == 'normal']
        important_features = ['duration', 'src_bytes', 'dst_bytes', 'count']
        
        feature_means = []
        feature_names = []
        for feat in important_features:
            if feat in normal_data.columns:
                feature_means.append(normal_data[feat].mean())
                feature_names.append(feat)
        
        plt.bar(feature_names, feature_means, color='lightgreen', alpha=0.7)
        plt.title('Promedios de Características\nen D1 (Normal)')
        plt.ylabel('Valor Promedio')
        plt.xticks(rotation=45)
        plt.yscale('log')
        
        # Subplot 8: Matriz de correlación para D3 (muestra)
        plt.subplot(3, 3, 8)
        # Seleccionar algunas características numéricas para correlación
        numeric_features = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']
        available_features = [f for f in numeric_features if f in attack_data.columns]
        
        if len(available_features) > 1:
            corr_matrix = attack_data[available_features].corr()
            im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(im, shrink=0.8)
            plt.xticks(range(len(available_features)), available_features, rotation=45)
            plt.yticks(range(len(available_features)), available_features)
            plt.title('Correlación de Características\nen D3 (Ataques)')
        
        # Subplot 9: Resumen de cobertura de ataques
        plt.subplot(3, 3, 9)
        train_attacks_set = set(train_df['attack'].unique())
        test_attacks_set = set(test_df['attack'].unique())
        
        only_train = len(train_attacks_set - test_attacks_set)
        only_test = len(test_attacks_set - train_attacks_set)
        common = len(train_attacks_set & test_attacks_set)
        
        labels = ['Solo\nEntrenamiento', 'Solo\nPrueba', 'Comunes']
        sizes = [only_train, only_test, common]
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Cobertura de Tipos\nde Ataques')
        
        plt.tight_layout()

        # Guardar la figura en disco en lugar de mostrarla (no bloquea la ejecución
        # y queda lista para la memoria del TFG)
        figures_dir = os.path.join(RAIZ_REPO, "Resultados", "figuras")
        os.makedirs(figures_dir, exist_ok=True)
        figure_path = os.path.join(figures_dir, 'eda_distribuciones_divisiones.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Figura del EDA guardada en: {figure_path}")


def main(aplicar_seleccion=True):
    """
    Función principal que ejecuta todo el pipeline especializado

    Parameters:
    -----------
    aplicar_seleccion : bool
        Si True (por defecto), aplica la selección de características 4.3.5
        y los CSVs de X se guardan ya filtrados. Si False (CLI:
        --sin-seleccion), los CSVs se regeneran con las 122 características
        completas y TODOS los artefactos de salida (CSVs, transformers.joblib,
        mappings y guía de uso) llevan el sufijo '_sin_seleccion', de modo que
        ambas variantes coexisten sin pisarse — útil para el experimento
        con/sin selección de la memoria (sección 4.3.5).
    """
    print("SISTEMA DE DETECCIÓN DE INTRUSIONES - NSL-KDD")
    print("PIPELINE ESPECIALIZADO PARA MODELOS DE ANOMALÍAS Y FIRMAS")
    print("=" * 80)
    
    # Crear instancia del preprocesador
    preprocessor = NSLKDDPreprocessor()
    
    # Paso 1: Cargar el dataset
    print("\n🔄 PASO 1: Cargando dataset...")
    train_df, test_df = preprocessor.load_dataset()
    
    if train_df is not None and test_df is not None:
        # Paso 2: Análisis exploratorio especializado
        print("\n🔍 PASO 2: Análisis exploratorio especializado...")
        train_df_analyzed, test_df_analyzed = preprocessor.exploratory_data_analysis(train_df, test_df)
        
        # Paso 3: Crear divisiones especializadas
        print("\n✂️  PASO 3: Creando divisiones especializadas...")
        data_splits = preprocessor.create_specialized_data_splits(train_df_analyzed, test_df_analyzed)
        
        # Paso 4: Preprocesar las divisiones
        print("\n🔧 PASO 4: Preprocesando divisiones especializadas...")
        processed_splits = preprocessor.preprocess_specialized_splits(data_splits, scaler_type='minmax')

        # Configurar directorio de salida. En modo sin selección, TODOS los
        # artefactos (CSVs, transformers.joblib, mappings, guía de uso) llevan
        # el sufijo '_sin_seleccion' para no pisar la variante por defecto.
        output_directory = os.path.join(RAIZ_REPO, "Resultados")
        sufijo_variante = '' if aplicar_seleccion else '_sin_seleccion'
        base_filename = f"{output_directory}/specialized_nsl_kdd{sufijo_variante}"

        # Crear directorio si no existe
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Paso 4b: Selección de características (4.3.5) — filtro varianza/correlación
        # sobre D1+D3 + importancias RF. Los CSVs se guardan YA filtrados; con
        # aplicar_seleccion=False se regeneran sin selección (experimento con/sin).
        if aplicar_seleccion:
            print("\n🧮 PASO 4b: Selección de características (4.3.5)...")
            preprocessor.select_features(processed_splits, output_dir=output_directory)
        else:
            # En esta variante NO se genera selected_features.txt (no habría
            # selección que documentar) y NO se toca el de la variante con
            # selección: los artefactos van bajo el sufijo '_sin_seleccion'.
            print("\n⏭️  PASO 4b OMITIDO: los CSVs se generan SIN selección de características")
            print(f"   ✓ Artefactos con sufijo '{sufijo_variante}' (la variante con selección no se pisa)")

        # Paso 5: Guardar todas las divisiones
        print("\n💾 PASO 5: Guardando divisiones especializadas...")

        # Guardar todas las divisiones
        saved_files = preprocessor.save_specialized_splits(data_splits, processed_splits, base_filename)
        
        # Paso 6: Mostrar resumen final y guía de uso
        print("\n" + "="*80)
        print("🎉 PIPELINE ESPECIALIZADO COMPLETADO EXITOSAMENTE")
        print("="*80)
        
        print(f"\n📊 RESUMEN DE DIVISIONES CREADAS:")
        print(f"   D1 - Normal para Anomalías:     {len(data_splits['D1_normal_for_anomaly']):,} instancias")
        print(f"   D2 - Test Completo:             {len(data_splits['D2_complete_test']):,} instancias") 
        print(f"   D3 - Ataques para Firmas:       {len(data_splits['D3_known_attacks_for_signatures']):,} instancias")
        
        print(f"\n🎯 PRÓXIMOS PASOS RECOMENDADOS:")
        print(f"   1. Usar D1 para entrenar modelos de anomalías (IsolationForest, OneClassSVM)")
        print(f"   2. Analizar D3 para extraer patrones/firmas de ataques conocidos")
        print(f"   3. Evaluar ambos modelos con D2 para obtener métricas realistas")
        print(f"   4. Combinar ambos enfoques para un sistema híbrido robusto")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   📍 Directorio: {output_directory}")
        print(f"   📄 Prefijo: specialized_nsl_kdd{sufijo_variante}_*")
        print(f"   📊 Total: {len(saved_files) + 2} archivos")
        
        print(f"\n💡 Para cargar y usar los datos, consulta:")
        print(f"   📖 {base_filename}_usage_guide.txt")
        print(f"   🗺️  {base_filename}_mappings_and_info.txt")
        
        return data_splits, processed_splits, preprocessor
    
    else:
        print("❌ No se pudo completar el pipeline debido a errores en la carga del dataset.")
        return None, None, None


# Función auxiliar para cargar divisiones procesadas
def load_specialized_splits(base_path='specialized_nsl_kdd'):
    """
    Función auxiliar para cargar las divisiones especializadas ya procesadas
    
    Parameters:
    -----------
    base_path : str
        Prefijo de los archivos guardados
        
    Returns:
    --------
    dict
        Diccionario con las divisiones cargadas
    """
    print("📂 CARGANDO DIVISIONES ESPECIALIZADAS...")
    
    try:
        splits = {}
        
        # Cargar D1 - Normal para anomalías
        splits['D1'] = {
            'X': pd.read_csv(f'{base_path}_processed_X_D1_normal_for_anomaly.csv'),
            'y_category': pd.read_csv(f'{base_path}_processed_y_category_D1_normal_for_anomaly.csv')
        }
        
        # Cargar D2 - Test completo  
        splits['D2'] = {
            'X': pd.read_csv(f'{base_path}_processed_X_D2_complete_test.csv'),
            'y_attack': pd.read_csv(f'{base_path}_processed_y_attack_D2_complete_test.csv'),
            'y_category': pd.read_csv(f'{base_path}_processed_y_category_D2_complete_test.csv')
        }
        
        # Cargar D3 - Ataques para firmas
        splits['D3'] = {
            'X': pd.read_csv(f'{base_path}_processed_X_D3_known_attacks_for_signatures.csv'),
            'y_attack': pd.read_csv(f'{base_path}_processed_y_attack_D3_known_attacks_for_signatures.csv'),
            'y_category': pd.read_csv(f'{base_path}_processed_y_category_D3_known_attacks_for_signatures.csv')
        }
        
        print("✅ Divisiones especializadas cargadas exitosamente")
        for split_name, split_data in splits.items():
            print(f"   {split_name}: X{split_data['X'].shape}")
        
        return splits
        
    except Exception as e:
        print(f"❌ Error al cargar divisiones: {e}")
        return None


# Ejemplo de uso completo
if __name__ == "__main__":
    # CLI: con --sin-seleccion se omite la selección 4.3.5 y todos los
    # artefactos se guardan con el sufijo '_sin_seleccion', de modo que las
    # variantes con y sin selección coexisten en Resultados/ sin pisarse.
    parser = argparse.ArgumentParser(
        description="Pipeline de preprocesamiento NSL-KDD: divisiones D1/D2/D3 "
                    "con selección de características 4.3.5 (por defecto)"
    )
    parser.add_argument(
        '--sin-seleccion', action='store_true',
        help="Omite la selección de características 4.3.5; los artefactos se "
             "guardan con el sufijo '_sin_seleccion' (122 features completas) "
             "y no pisan los de la variante por defecto"
    )
    args = parser.parse_args()

    # Ejecutar el pipeline especializado completo
    data_splits, processed_splits, preprocessor = main(aplicar_seleccion=not args.sin_seleccion)
    
    if data_splits is not None:
        print(f"\n🔍 VERIFICACIÓN FINAL DE DIVISIONES:")
        print(f"   D1 contiene solo 'normal': {(data_splits['D1_normal_for_anomaly']['attack'] == 'normal').all()}")
        print(f"   D2 contiene normal + ataques: {len(data_splits['D2_complete_test']['attack'].unique())} tipos únicos")
        print(f"   D3 contiene solo ataques: {(data_splits['D3_known_attacks_for_signatures']['attack'] != 'normal').all()}")
        
        print(f"\n✨ ¡LAS DIVISIONES ESTÁN LISTAS PARA USAR!")


"""
EJEMPLO DE USO POSTERIOR:

# Cargar divisiones ya procesadas
splits = load_specialized_splits('C:/ruta/specialized_nsl_kdd')

# Entrenar modelo de anomalías con D1
from sklearn.ensemble import IsolationForest
anomaly_model = IsolationForest(contamination=0.1, random_state=42)
anomaly_model.fit(splits['D1']['X'])  # Solo datos normales

# Extraer patrones de D3 para firmas
attack_patterns = {}
for category in splits['D3']['y_category']['category_original'].unique():
    mask = splits['D3']['y_category']['category_original'] == category
    category_data = splits['D3']['X'][mask]
    attack_patterns[category] = {
        'mean': category_data.mean(),
        'std': category_data.std(),
        'count': len(category_data)
    }

# Evaluar en D2
predictions = anomaly_model.predict(splits['D2']['X'])
# Analizar resultados...
"""