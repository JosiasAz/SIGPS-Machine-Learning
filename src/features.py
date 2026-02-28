from dataclasses import dataclass
import numpy as np

ORDEM_FEATURES = [
    "urgencia",
    "historico_consultas",
    "historico_faltas",
    "minutos_na_fila",
]

@dataclass(frozen=True)
class FeaturesPrioridade:
    urgencia: int
    historico_consultas: int
    historico_faltas: int
    minutos_na_fila: int

    def para_array(self) -> np.ndarray:
        
        '''
        traduz o objeto (que é legível para humanos) para um formato matemático (que é o que a máquina entende).
        '''
        return np.array(
            [
                self.urgencia,
                self.historico_consultas,
                self.historico_faltas,
                self.minutos_na_fila,
            ],
            dtype=np.float32,
        )
    
def para_entrada_modelo(features: FeaturesPrioridade) -> np.ndarray:
    '''
    transforma o array de 1 dimensão em um array de 2 dimensões.
    '''
    return features.para_array().reshape(1, -1)