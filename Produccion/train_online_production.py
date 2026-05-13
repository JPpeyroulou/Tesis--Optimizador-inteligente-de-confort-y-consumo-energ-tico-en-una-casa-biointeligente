#!/usr/bin/env python3
"""
Sistema de Aprendizaje Online en Producción
Entrenamiento en tiempo real con entornos de producción personalizados
"""

import os
import sys
import yaml
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import warnings

# ============ REGISTRAR ENTORNO PERSONALIZADO ============
# Importar y registrar antes de cualquier otra cosa
import sys
import os

# Añadir el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar el archivo de registro (ejecuta el registro automáticamente)
try:
    from register_production_envs import *
    print("✅ Entorno de producción registrado")
except ImportError as e:
    print(f"⚠️  No se pudo registrar el entorno: {e}")
    # Intentar crear el entorno directamente
    pass

# ============ IMPORTS PRINCIPALES ============
import gymnasium as gym
import sinergym
from sinergym.utils.common import create_environment as create_sinergym_env, import_from_path
from sinergym.utils.wrappers import LoggerWrapper, NormalizeObservation, MultiObsWrapper
from sinergym.utils.rewards import NuestroRewardMultizona

# Stable Baselines3
try:
    from stable_baselines3 import SAC, PPO, TD3, DDPG
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️  Stable Baselines3 no encontrado")

# ============ CONFIGURACIÓN ============
def load_config(config_path: str) -> Dict[str, Any]:
    """Carga configuración desde archivo YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], path: str):
    """Guarda configuración a archivo"""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

# ============ CREACIÓN DE ENTORNO ============
def create_production_env(config: Dict[str, Any]) -> gym.Env:
    """Crea el entorno de producción"""
    env_name = config['environment']
    
    print(f"\n🎯 Creando entorno: {env_name}")
    print(f"   Tipo: {config.get('env_type', 'production')}")
    
    # Verificar si es un entorno de producción personalizado
    if 'multizona-production' in env_name or env_name == 'Eplus-pyenv-multizona-production-v1':
        # ===== ENTORNO DE PRODUCCIÓN PERSONALIZADO =====
        try:
            # Importar nuestra clase personalizada
            from pyenv_production import PyEnvProduction
            
            # Obtener configuración del entorno
            env_config = config.get('env_config', {})
            
            # Configuración por defecto para producción
            default_config = {
                'building_file': 'OfficeMedium_Zone_4.pkl',
                'weather_files': ['USA_CO_Denver.Intl.AP.725650_TMY3.epw'],
                # IMPORTANTE: variables y meters deben venir del YAML, no del default
                # Dejarlos vacíos para evitar mezclar con las del YAML
                'variables': {},
                'meters': {},
                'actuators': {
                    'heating_setpoint_1': ('Schedule:Compact', 'Schedule Value', 'Heating Setpoint 1'),
                    'cooling_setpoint_1': ('Schedule:Compact', 'Schedule Value', 'Cooling Setpoint 1'),
                    'heating_setpoint_2': ('Schedule:Compact', 'Schedule Value', 'Heating Setpoint 2'),
                    'cooling_setpoint_2': ('Schedule:Compact', 'Schedule Value', 'Cooling Setpoint 2'),
                    'heating_setpoint_3': ('Schedule:Compact', 'Schedule Value', 'Heating Setpoint 3'),
                    'cooling_setpoint_3': ('Schedule:Compact', 'Schedule Value', 'Cooling Setpoint 3'),
                    'heating_setpoint_4': ('Schedule:Compact', 'Schedule Value', 'Heating Setpoint 4'),
                    'cooling_setpoint_4': ('Schedule:Compact', 'Schedule Value', 'Cooling Setpoint 4')
                },
                'action_space': gym.spaces.Box(
                    low=np.array([-20.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0], dtype=np.float32),
                    high=np.array([30.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0], dtype=np.float32),
                    dtype=np.float32
                ),
                'reward': NuestroRewardMultizona,  # ✅ Usar NuestroRewardMultizona por defecto
                'reward_kwargs': {
                    'temperature_variables': ['zone_air_temperature_1', 'zone_air_temperature_2', 
                                            'zone_air_temperature_3', 'zone_air_temperature_4'],
                    'humidity_variables': [],  # Agregar si es necesario
                    'energy_variables': ['total_electric_demand'],  # Debe ser lista
                    'energy_weight': 0.5,
                    'lambda_energy': 1.0,
                    'lambda_temperature': 30,
                    'high_price': 14.493,
                    'low_price': 4.556,
                    'schedule_csv': None
                },
                'env_name': env_name,
                'time_variables': ['month', 'day_of_month', 'hour'],  # ✅ Necesario para NuestroRewardMultizona
                'config_params': {
                    'timesteps_per_hour': 12,  # 12 pasos por hora (cada paso = 5 minutos)
                    'runperiod': (1, 1, 12, 31),
                    'action_definition': {}
                },
                'production_config': {
                    'data_mode': 'homeassistant',  # Cambiado a homeassistant por defecto
                    'action_delay': 1.0,
                    'max_steps_per_episode': 288,  # 24 horas = 288 pasos (12 pasos/hora, 5 min/paso)
                    'safety_limits': {
                        'max_zone_temperature': 28.0,
                        'min_zone_temperature': 16.0
                    }
                }
            }
            
            # Combinar configuración por defecto con la proporcionada
            import copy
            final_config = copy.deepcopy(default_config)
            
            # Actualizar recursivamente, pero para variables y meters, reemplazar completamente
            def deep_update(d, u):
                for k, v in u.items():
                    # Para variables y meters, reemplazar completamente (no merge)
                    if k in ['variables', 'meters'] and isinstance(v, dict):
                        d[k] = copy.deepcopy(v)
                    # Para production_config, hacer merge profundo pero preservar data_mode del YAML
                    elif k == 'production_config' and isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        # Preservar data_mode del YAML si existe
                        yaml_data_mode = v.get('data_mode')
                        deep_update(d[k], v)
                        if yaml_data_mode:
                            d[k]['data_mode'] = yaml_data_mode
                    elif isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        deep_update(d[k], v)
                    else:
                        d[k] = v
            
            deep_update(final_config, env_config)
            
            # Procesar reward si viene como string (desde YAML)
            if isinstance(final_config.get('reward'), str):
                try:
                    final_config['reward'] = import_from_path(final_config['reward'])
                    print(f"   ✅ Recompensa importada: {final_config['reward'].__name__}")
                except Exception as e:
                    print(f"   ⚠️  Error importando recompensa: {e}, usando NuestroRewardMultizona")
                    final_config['reward'] = NuestroRewardMultizona
            
            # Procesar action_space si viene como dict (desde YAML)
            if isinstance(final_config.get('action_space'), dict):
                action_space_dict = final_config['action_space']
                if 'low' in action_space_dict and 'high' in action_space_dict:
                    final_config['action_space'] = gym.spaces.Box(
                        low=np.array(action_space_dict['low'], dtype=np.float32),
                        high=np.array(action_space_dict['high'], dtype=np.float32),
                        dtype=np.float32
                    )
                    print(f"   ✅ Action space procesado desde YAML: {len(action_space_dict['low'])} dimensiones")
                elif isinstance(final_config.get('action_space'), str):
                    # Si viene como string (código Python), evaluarlo
                    final_config['action_space'] = eval(final_config['action_space'])
                    print(f"   ✅ Action space procesado desde YAML (string): {final_config['action_space'].shape[0]} dimensiones")
            
            # Procesar variables si vienen en formato YAML (lista [energyplus_var_name, key])
            # Convertir al formato Sinergym que espera PyEnvProduction
            if isinstance(final_config.get('variables'), dict):
                print(f"\n🔍 DEBUG: Procesando variables del YAML...")
                print(f"   Variables en YAML (antes de procesar): {list(final_config['variables'].keys())}")
                # IMPORTANTE: Limpiar variables_processed para evitar mezclar con defaults
                variables_processed = {}
                
                for var_name, var_spec in final_config['variables'].items():
                    if isinstance(var_spec, list) and len(var_spec) >= 2:
                        # Formato YAML: [energyplus_var_name, key]
                        # Convertir a formato Sinergym: {energyplus_var_name: {'variable_names': var_name, 'keys': key}}
                        eplus_var_name = var_spec[0]
                        key = var_spec[1]
                        
                        # Si ya existe esta variable de EnergyPlus, agregar a la lista de keys
                        if eplus_var_name in variables_processed:
                            existing_keys = variables_processed[eplus_var_name]['keys']
                            if isinstance(existing_keys, list):
                                existing_keys.append(key)
                            else:
                                variables_processed[eplus_var_name]['keys'] = [existing_keys, key]
                        else:
                            variables_processed[eplus_var_name] = {
                                'variable_names': var_name,  # Nombre que se usará en observaciones
                                'keys': key
                            }
                    
                    elif isinstance(var_spec, tuple) and len(var_spec) >= 2:
                        # Formato tupla: (energyplus_var_name, key)
                        eplus_var_name = var_spec[0]
                        key = var_spec[1]
                        
                        if eplus_var_name in variables_processed:
                            existing_keys = variables_processed[eplus_var_name]['keys']
                            if isinstance(existing_keys, list):
                                existing_keys.append(key)
                            else:
                                variables_processed[eplus_var_name]['keys'] = [existing_keys, key]
                        else:
                            variables_processed[eplus_var_name] = {
                                'variable_names': var_name,
                                'keys': key
                            }
                    
                    elif isinstance(var_spec, dict):
                        # Ya está en formato Sinergym
                        variables_processed[var_name] = var_spec
                    else:
                        # Mantener como está
                        variables_processed[var_name] = var_spec
                
                final_config['variables'] = variables_processed
                print(f"   ✅ Variables procesadas: {len(variables_processed)} variables")
                print(f"   Variables finales: {list(variables_processed.keys())}")
            
            # Procesar meters: Sinergym invierte el formato (EnergyPlus name -> variable name)
            # Formato YAML: "Heat Pump:Heating:Electricity: total_electricity_HVAC"
            # Formato esperado: {"total_electricity_HVAC": "Heat Pump:Heating:Electricity"}
            if 'meters' in final_config and isinstance(final_config['meters'], dict):
                print(f"\n🔍 DEBUG: Procesando meters del YAML...")
                print(f"   Meters en YAML (antes de procesar): {final_config['meters']}")
                # IMPORTANTE: Limpiar meters_processed para evitar mezclar con defaults
                meters_processed = {}
                for eplus_name, var_name in final_config['meters'].items():
                    # Si el formato es invertido (como en nuestroMultrizona.yaml)
                    # eplus_name = "Heat Pump:Heating:Electricity", var_name = "total_electricity_HVAC"
                    meters_processed[var_name] = eplus_name
                final_config['meters'] = meters_processed
                print(f"   ✅ Meters procesados: {len(meters_processed)} meters")
                print(f"   Meters finales: {meters_processed}")
            
            # Crear entorno directamente (sin gym.make para más control)
            env = PyEnvProduction(
                building_file=final_config['building_file'],
                weather_files=final_config['weather_files'],
                variables=final_config['variables'],
                meters=final_config['meters'],
                actuators=final_config['actuators'],
                action_space=final_config['action_space'],
                reward=final_config['reward'],
                reward_kwargs=final_config['reward_kwargs'],
                env_name=final_config['env_name'],
                config_params=final_config['config_params'],
                production_config=final_config['production_config'],
                time_variables=final_config.get('time_variables', ['month', 'day_of_month', 'hour'])  # ✅ Necesario para NuestroRewardMultizona
            )
            
            print(f"✅ Entorno de producción creado:")
            print(f"   - Variables: {len(env.variable_names)}")
            print(f"   - Modo datos: {env.production_config.get('data_mode', 'N/A')}")
            # Verificar que action_space sea un objeto Box antes de acceder a shape
            if hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0:
                print(f"   - Acciones: {env.action_space.shape[0]} dimensiones")
            else:
                print(f"   - Acciones: {env.action_space}")
            
            return env
            
        except Exception as e:
            print(f"❌ Error creando entorno de producción: {e}")
            raise
    
    else:
        # ===== ENTORNO NORMAL DE SINERGYM =====
        try:
            # Usar la función estándar de Sinergym
            env = create_sinergym_env(
                id_env=env_name,
                **config.get('env_kwargs', {})
            )
            print(f"✅ Entorno Sinergym creado: {env_name}")
            return env
        except KeyError:
            # Intentar con gym.make
            try:
                env = gym.make(env_name)
                print(f"✅ Entorno Gymnasium creado: {env_name}")
                return env
            except Exception as e:
                print(f"❌ Error creando entorno {env_name}: {e}")
                raise

def apply_wrappers(env: gym.Env, config: Dict[str, Any]) -> gym.Env:
    """Aplica wrappers al entorno"""
    wrappers = config.get('wrappers', [])
    
    print(f"\n🎁 Aplicando wrappers ({len(wrappers)} configurados):")
    
    # Lista de wrappers ya aplicados para evitar duplicados
    applied_wrappers = set()
    
    # Aplicar wrappers personalizados desde configuración
    for wrapper_cfg in wrappers:
        wrapper_name = wrapper_cfg['name']
        wrapper_params = wrapper_cfg.get('params', {})
        
        # Evitar aplicar el mismo wrapper dos veces
        if wrapper_name in applied_wrappers:
            print(f"   ⚠️  {wrapper_name} ya aplicado, omitiendo")
            continue
        
        try:
            # Importar dinámicamente
            module = __import__('sinergym.utils.wrappers', fromlist=[wrapper_name])
            wrapper_class = getattr(module, wrapper_name)
            
            # LoggerWrapper no acepta parámetros como 'flag', solo storage_class
            if wrapper_name == 'LoggerWrapper':
                # Remover parámetros no válidos
                wrapper_params = {k: v for k, v in wrapper_params.items() if k == 'storage_class'}
                if wrapper_params:
                    env = wrapper_class(env, **wrapper_params)
                else:
                    env = wrapper_class(env)
            else:
                # Aplicar wrapper con parámetros
                env = wrapper_class(env, **wrapper_params)
            
            applied_wrappers.add(wrapper_name)
            print(f"   ✅ {wrapper_name}")
            
        except Exception as e:
            print(f"   ⚠️  No se pudo aplicar {wrapper_name}: {e}")
    
    # Wrapper de normalización de observaciones (si no está en la lista y está habilitado)
    if config.get('normalize', False) and 'NormalizeObservation' not in applied_wrappers:
        env = NormalizeObservation(env)
        print("   ✅ NormalizeObservation")
    
    # Wrapper de normalización de acciones (para que coincida con el modelo entrenado)
    # IMPORTANTE: Aplicar NormalizeAction para que el action_space sea [-1, 1] como el modelo
    # Esto es CRÍTICO para que el modelo pueda entrenarse online
    if 'NormalizeAction' not in applied_wrappers:
        try:
            from sinergym.utils.wrappers import NormalizeAction
            # Verificar que el action_space sea Box antes de aplicar
            if isinstance(env.action_space, gym.spaces.Box):
                env = NormalizeAction(env, normalize_range=(-1.0, 1.0))
                print("   ✅ NormalizeAction (action_space normalizado a [-1, 1])")
                print(f"      Action space original: {env.real_space if hasattr(env, 'real_space') else 'N/A'}")
                print(f"      Action space normalizado: {env.action_space}")
                applied_wrappers.add('NormalizeAction')
            else:
                print(f"   ⚠️  Action space no es Box, no se puede aplicar NormalizeAction")
        except ImportError:
            print("   ⚠️  NormalizeAction no disponible, el action_space puede no coincidir con el modelo")
        except Exception as e:
            print(f"   ⚠️  No se pudo aplicar NormalizeAction: {e}")
            import traceback
            traceback.print_exc()
    
    # Wrapper de logger (si no está en la lista y está habilitado)
    if config.get('logging', True) and 'LoggerWrapper' not in applied_wrappers:
        env = LoggerWrapper(env)
        print(f"   ✅ LoggerWrapper (logs en: {env.get_wrapper_attr('workspace_path')})")
    
    return env

# ============ DESNORMALIZACIÓN DE ACCIONES ============
def adapt_action_dimensions(action: np.ndarray, target_dim: int, 
                           target_low: np.ndarray, target_high: np.ndarray,
                           verbose: bool = False) -> np.ndarray:
    """
    Adapta una acción a la dimensión objetivo.
    
    Args:
        action: Acción del modelo
        target_dim: Dimensión objetivo
        target_low: Límites inferiores del espacio objetivo
        target_high: Límites superiores del espacio objetivo
        verbose: Si True, imprime información
        
    Returns:
        Acción adaptada a la dimensión objetivo
    """
    current_dim = action.shape[0] if len(action.shape) > 0 else len(action)
    
    if current_dim == target_dim:
        return action
    elif current_dim > target_dim:
        # Reducir: tomar las primeras target_dim dimensiones
        if verbose:
            print(f"      ⚠️  Reduciendo acción de {current_dim} a {target_dim} dimensiones")
        return action[:target_dim]
    else:
        # Aumentar: necesitamos mapear las dimensiones del modelo a las del entorno
        if verbose:
            print(f"      ⚠️  Expandindo acción de {current_dim} a {target_dim} dimensiones")
            print(f"      ⚠️  ADVERTENCIA: No hay mapeo definido para dimensiones faltantes")
            print(f"      ⚠️  Las dimensiones {current_dim}-{target_dim-1} se rellenarán con valores por defecto")
            print(f"      ⚠️  Esto es INCORRECTO - necesitas definir el mapeo en la configuración")
        
        adapted = np.zeros(target_dim, dtype=action.dtype)
        
        # Mapeo básico: usar las primeras dimensiones del modelo para las primeras del entorno
        # Esto es solo un placeholder - DEBES definir el mapeo correcto según tus actuadores
        min_dim = min(current_dim, target_dim)
        adapted[:min_dim] = action[:min_dim]
        
        # Rellenar las dimensiones faltantes con valores por defecto (punto medio)
        # ⚠️ ESTO ES INCORRECTO - solo es un placeholder
        for i in range(min_dim, target_dim):
            default_value = (target_low[i] + target_high[i]) / 2.0
            adapted[i] = default_value
            if verbose:
                print(f"         ⚠️  Dimensión {i}: rellenada con {default_value:.1f} (VALOR POR DEFECTO - INCORRECTO)")
        
        if verbose:
            print(f"\n      ❌ ERROR: El modelo tiene {current_dim} dimensiones pero el entorno necesita {target_dim}")
            print(f"      ❌ Las dimensiones {min_dim}-{target_dim-1} están rellenas con valores por defecto")
            print(f"      ❌ DEBES definir un mapeo correcto en la configuración o usar el mismo action_space")
        
        return adapted

def denormalize_action(action: np.ndarray, action_space: gym.spaces.Box, 
                      model_action_space: Optional[gym.spaces.Box] = None,
                      verbose: bool = False) -> np.ndarray:
    """
    Desnormaliza una acción del rango del modelo al rango real del action_space.
    
    Args:
        action: Acción del modelo (puede estar normalizada)
        action_space: Espacio de acciones del entorno con los rangos reales
        model_action_space: Espacio de acciones del modelo (opcional, para mapeo directo)
        verbose: Si True, imprime información de la desnormalización
        
    Returns:
        Acción desnormalizada en el rango real
    """
    if not isinstance(action_space, gym.spaces.Box):
        return action
    
    action = np.array(action, dtype=np.float32)
    target_low = action_space.low
    target_high = action_space.high
    target_dim = action_space.shape[0]
    action_dim = action.shape[0] if len(action.shape) > 0 else len(action)
    
    # Verificar y adaptar dimensiones si es necesario
    if action_dim != target_dim:
        if verbose:
            print(f"      ⚠️  Dimensiones no coinciden: modelo={action_dim}, entorno={target_dim}")
        action = adapt_action_dimensions(action, target_dim, target_low, target_high, verbose)
    
    # Inicializar action_real como None para indicar que aún no se ha procesado
    action_real = None
    
    # Si tenemos el action_space del modelo, intentar mapear directamente
    if model_action_space is not None and isinstance(model_action_space, gym.spaces.Box):
        source_low = model_action_space.low
        source_high = model_action_space.high
        
        # Verificar que las dimensiones coincidan después de la adaptación
        if len(source_low) == len(action) and len(target_low) == len(action) and len(source_high) == len(action):
            # Mapear desde el rango del modelo al rango del entorno
            action_real = target_low + (action - source_low) / (source_high - source_low + 1e-8) * (target_high - target_low)
            
            if verbose:
                print(f"      🔄 Mapeando desde modelo [{np.min(source_low):.1f}, {np.max(source_high):.1f}]")
                print(f"         a entorno [{np.min(target_low):.1f}, {np.max(target_high):.1f}]")
        else:
            # Si las dimensiones no coinciden, usar detección automática
            if verbose:
                print(f"      ⚠️  Dimensiones no coinciden para mapeo directo (modelo={len(source_low)}, acción={len(action)}, entorno={len(target_low)})")
                print(f"      🔄 Usando detección automática de rango")
            # Continuar con la detección automática más abajo
            action_real = None
    
    # Si no se pudo mapear directamente, usar detección automática
    if action_real is None:
        # Detectar automáticamente el rango de la acción
        action_min = np.min(action)
        action_max = np.max(action)
        
        if verbose:
            print(f"      📊 Rango acción recibida: [{action_min:.3f}, {action_max:.3f}]")
            print(f"      📊 Rango esperado: [{np.min(target_low):.1f}, {np.max(target_high):.1f}]")
        
        # Verificar si ya está en el rango correcto (con tolerancia)
        tolerance = 0.1
        if np.all(action >= target_low - tolerance) and np.all(action <= target_high + tolerance):
            if verbose:
                print(f"      ✅ Acción ya está en rango real, sin cambios")
            return np.clip(action, target_low, target_high).astype(np.float32)
        
        # Si la acción está en [-1, 1], desnormalizar a [target_low, target_high]
        if action_min >= -1.1 and action_max <= 1.1:
            # Normalización [-1, 1] -> [target_low, target_high]
            action_real = target_low + (action + 1.0) / 2.0 * (target_high - target_low)
            if verbose:
                print(f"      🔄 Desnormalizando desde [-1, 1] a rango real")
        # Si la acción está en [0, 1], desnormalizar a [target_low, target_high]
        elif action_min >= -0.1 and action_max <= 1.1:
            # Normalización [0, 1] -> [target_low, target_high]
            action_real = target_low + action * (target_high - target_low)
            if verbose:
                print(f"      🔄 Desnormalizando desde [0, 1] a rango real")
        else:
            # Por defecto, asumir que está en [-1, 1]
            action_real = target_low + (action + 1.0) / 2.0 * (target_high - target_low)
            if verbose:
                print(f"      🔄 Desnormalizando (asumiendo [-1, 1]) a rango real")
    
    # Asegurar que esté dentro de los límites
    action_real = np.clip(action_real, target_low, target_high)
    
    if verbose:
        print(f"      ✅ Acción final: {action_real}")
    
    return action_real.astype(np.float32)

# ============ ADAPTACIÓN DE OBSERVACIONES ============
class ObservationAdapterWrapper(gym.ObservationWrapper):
    """Wrapper que adapta observaciones a la dimensión esperada por el modelo"""
    
    def __init__(self, env: gym.Env, target_dim: int):
        super().__init__(env)
        self.target_dim = target_dim
        current_dim = env.observation_space.shape[0]
        
        # Actualizar espacio de observación
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low[0] if current_dim > 0 else -np.inf,
            high=env.observation_space.high[0] if current_dim > 0 else np.inf,
            shape=(target_dim,),
            dtype=env.observation_space.dtype
        )
        
        print(f"   🔄 ObservationAdapter: {current_dim} → {target_dim} dimensiones")
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Adapta la observación a la dimensión objetivo"""
        current_dim = obs.shape[0] if len(obs.shape) > 0 else len(obs)
        
        if current_dim == self.target_dim:
            return obs
        elif current_dim > self.target_dim:
            # Reducir: tomar las primeras target_dim dimensiones
            return obs[:self.target_dim]
        else:
            # Aumentar: rellenar con ceros
            adapted = np.zeros(self.target_dim, dtype=obs.dtype)
            adapted[:current_dim] = obs
            return adapted

def adapt_observation(obs: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Adapta una observación a la dimensión esperada por el modelo.
    (Función auxiliar para uso manual)
    """
    current_dim = obs.shape[0] if len(obs.shape) > 0 else len(obs)
    
    if current_dim == target_dim:
        return obs
    elif current_dim > target_dim:
        return obs[:target_dim]
    else:
        adapted = np.zeros(target_dim, dtype=obs.dtype)
        adapted[:current_dim] = obs
        return adapted

# ============ CARGA DE MODELO ============
def load_production_model(model_path: str, config: Dict[str, Any], env: gym.Env) -> Tuple[Any, gym.Env]:
    """
    Carga un modelo preentrenado y adapta el entorno si es necesario.
    
    Returns:
        Tuple[model, env]: Modelo cargado y entorno (posiblemente adaptado)
    """
    """Carga un modelo preentrenado para producción"""
    print(f"\n🤖 Cargando modelo desde: {model_path}")
    
    if not HAS_SB3:
        print("❌ Stable Baselines3 no está instalado")
        return None
    
    # Determinar tipo de algoritmo
    algo_config = config.get('algorithm', {})
    algo_name = algo_config.get('name', 'SAC')
    
    # Verificar espacios antes de cargar
    try:
        # Intentar cargar sin env primero para ver los espacios del modelo
        if algo_name == 'SAC':
            temp_model = SAC.load(model_path, device='cpu')
        elif algo_name == 'PPO':
            temp_model = PPO.load(model_path, device='cpu')
        elif algo_name == 'TD3':
            temp_model = TD3.load(model_path, device='cpu')
        elif algo_name == 'DDPG':
            temp_model = DDPG.load(model_path, device='cpu')
        else:
            temp_model = None
        
        if temp_model:
            model_obs_space = temp_model.observation_space
            env_obs_space = env.observation_space
            
            print(f"\n📊 Espacios de observación:")
            print(f"   Modelo: {model_obs_space}")
            print(f"   Entorno: {env_obs_space}")
            
            # DEBUG: Desglose detallado
            print(f"\n📊 DESGLOSE DEL MODELO:")
            print(f"   Dimensiones: {model_obs_space.shape[0]}")
            print(f"   Shape: {model_obs_space.shape}")
            
            print(f"\n📊 DESGLOSE DEL ENTORNO:")
            print(f"   Dimensiones: {env_obs_space.shape[0]}")
            print(f"   Shape: {env_obs_space.shape}")
            
            if hasattr(env, 'time_variables'):
                print(f"   time_variables: {len(env.time_variables)} = {env.time_variables}")
            if hasattr(env, 'variable_names'):
                print(f"   variable_names: {len(env.variable_names)} = {env.variable_names}")
            if hasattr(env, 'meter_names'):
                print(f"   meter_names: {len(env.meter_names)} = {env.meter_names}")
            
            if model_obs_space.shape != env_obs_space.shape:
                print(f"\n⚠️  ADVERTENCIA: Los espacios de observación no coinciden!")
                print(f"   Modelo espera: {model_obs_space.shape[0]} dimensiones")
                print(f"   Entorno tiene: {env_obs_space.shape[0]} dimensiones")
                print(f"\n   🔄 Aplicando ObservationAdapterWrapper para adaptar observaciones...")
                
                # Aplicar wrapper de adaptación ANTES de cargar el modelo
                target_dim = model_obs_space.shape[0]
                env = ObservationAdapterWrapper(env, target_dim)
                print(f"   ✅ Entorno adaptado: {env_obs_space.shape[0]} → {target_dim} dimensiones")
                
                # Cargar sin env para evitar verificación estricta
                if algo_name == 'SAC':
                    model = SAC.load(model_path, device='cpu')
                elif algo_name == 'PPO':
                    model = PPO.load(model_path, device='cpu')
                elif algo_name == 'TD3':
                    model = TD3.load(model_path, device='cpu')
                elif algo_name == 'DDPG':
                    model = DDPG.load(model_path, device='cpu')
                
                # Actualizar el entorno del modelo (ahora ya está adaptado)
                model.set_env(env)
                print(f"   ✅ Modelo cargado y entorno adaptado correctamente")
            else:
                # Los espacios coinciden, cargar normalmente
                if algo_name == 'SAC':
                    model = SAC.load(model_path, env=env, verbose=1)
                elif algo_name == 'PPO':
                    model = PPO.load(model_path, env=env, verbose=1)
                elif algo_name == 'TD3':
                    model = TD3.load(model_path, env=env, verbose=1)
                elif algo_name == 'DDPG':
                    model = DDPG.load(model_path, env=env, verbose=1)
                print(f"   ✅ Espacios coinciden, modelo cargado correctamente")
        else:
            # Fallback: cargar normalmente
            if algo_name == 'SAC':
                model = SAC.load(model_path, env=env, verbose=1)
            elif algo_name == 'PPO':
                model = PPO.load(model_path, env=env, verbose=1)
            elif algo_name == 'TD3':
                model = TD3.load(model_path, env=env, verbose=1)
            elif algo_name == 'DDPG':
                model = DDPG.load(model_path, env=env, verbose=1)
    
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️  Error verificando espacios: {e}")
        
        # Cargar modelo sin asignar entorno para evitar verificación estricta
        if algo_name == 'SAC':
            model = SAC.load(model_path, device='cpu')
        elif algo_name == 'PPO':
            model = PPO.load(model_path, device='cpu')
        elif algo_name == 'TD3':
            model = TD3.load(model_path, device='cpu')
        elif algo_name == 'DDPG':
            model = DDPG.load(model_path, device='cpu')
        
        model_obs_dim = model.observation_space.shape[0]
        env_obs_dim = env.observation_space.shape[0]
        model_action_space = model.action_space
        env_action_space = env.action_space
        
        # Verificar si el problema es solo de action space (normalizado vs real)
        action_space_mismatch = (
            "Action spaces do not match" in error_msg or
            model_action_space.shape != env_action_space.shape or
            not np.allclose(model_action_space.low, env_action_space.low) or
            not np.allclose(model_action_space.high, env_action_space.high)
        )
        
        # Verificar si el problema es de observation space
        obs_space_mismatch = (
            "Observation spaces do not match" in error_msg or
            model_obs_dim != env_obs_dim
        )
        
        if action_space_mismatch and not obs_space_mismatch:
            print(f"   ℹ️  Action spaces diferentes (normalizado vs real) - esto es esperado")
            print(f"      Modelo: {model_action_space}")
            print(f"      Entorno: {env_action_space}")
            print(f"      Las acciones se desnormalizarán automáticamente durante la ejecución")
        
        if obs_space_mismatch:
            print(f"   ⚠️  El modelo espera {model_obs_dim} dimensiones")
            print(f"   ⚠️  El entorno proporciona {env_obs_dim} dimensiones")
            print(f"   🔄 Aplicando wrapper de adaptación de observaciones...")
            
            # Aplicar wrapper de adaptación
            env = ObservationAdapterWrapper(env, model_obs_dim)
            print(f"   ✅ Entorno adaptado: {env_obs_dim} → {model_obs_dim} dimensiones")
        
        # Intentar asignar el entorno (puede fallar si los action spaces no coinciden, pero está bien)
        env_assigned = False
        try:
            model.set_env(env)
            print(f"   ✅ Entorno asignado al modelo")
            env_assigned = True
        except Exception as e2:
            error_msg = str(e2)
            if "Action spaces do not match" in error_msg:
                print(f"   ℹ️  No se pudo asignar entorno (action spaces diferentes) - intentando con VecEnv")
            elif "Observation spaces do not match" in error_msg:
                print(f"   ℹ️  No se pudo asignar entorno (observation spaces diferentes) - intentando con VecEnv")
            else:
                print(f"   ⚠️  No se pudo asignar entorno: {e2}")
                print(f"   ⚠️  Intentando con VecEnv...")
            
            # Para entrenamiento, necesitamos asignar el entorno de otra manera
            # Usar VecEnv directamente si es necesario
            try:
                from stable_baselines3.common.vec_env import DummyVecEnv
                vec_env = DummyVecEnv([lambda: env])
                model.set_env(vec_env)
                print(f"   ✅ Entorno asignado usando VecEnv")
                env_assigned = True
            except Exception as e3:
                error_msg3 = str(e3)
                # Si el problema es de action space, verificar si NormalizeAction está aplicado
                if "Action spaces do not match" in error_msg3:
                    print(f"   ⚠️  Action space aún no coincide después de VecEnv")
                    print(f"   ℹ️  Verificando si NormalizeAction está aplicado...")
                    # Verificar si el entorno tiene NormalizeAction
                    unwrapped = env
                    has_normalize_action = False
                    while hasattr(unwrapped, 'env'):
                        if hasattr(unwrapped, 'normalized_space'):
                            has_normalize_action = True
                            print(f"   ✅ NormalizeAction detectado en el wrapper")
                            break
                        unwrapped = unwrapped.env
                    
                    if not has_normalize_action:
                        print(f"   ⚠️  NormalizeAction no está aplicado - intentando aplicar ahora...")
                        try:
                            from sinergym.utils.wrappers import NormalizeAction
                            env = NormalizeAction(env, normalize_range=(-1.0, 1.0))
                            print(f"   ✅ NormalizeAction aplicado")
                            # Intentar asignar de nuevo
                            vec_env = DummyVecEnv([lambda: env])
                            model.set_env(vec_env)
                            print(f"   ✅ Entorno asignado después de aplicar NormalizeAction")
                            env_assigned = True
                        except Exception as e4:
                            print(f"   ⚠️  No se pudo aplicar NormalizeAction: {e4}")
                            print(f"   ⚠️  El modelo se usará sin entorno asignado (solo para testing)")
                    else:
                        print(f"   ⚠️  NormalizeAction está aplicado pero aún hay problemas")
                        print(f"   ⚠️  El modelo se usará sin entorno asignado (solo para testing)")
                else:
                    print(f"   ⚠️  No se pudo asignar entorno con VecEnv: {e3}")
                    print(f"   ⚠️  El modelo se usará sin entorno asignado (solo para testing)")
        
        # Guardar flag para verificar antes de entrenar
        if not env_assigned:
            print(f"   ⚠️  ADVERTENCIA: El entorno no pudo ser asignado al modelo")
            print(f"   ⚠️  El entrenamiento fallará. Solo se puede usar para testing.")
        
        return model, env
    else:
        # Por defecto, SAC
        print(f"⚠️  Algoritmo {algo_name} no reconocido, usando SAC")
        try:
            model = SAC.load(model_path, env=env, verbose=1)
        except Exception as e:
            print(f"   ⚠️  Error cargando con entorno: {e}")
            model = SAC.load(model_path, device='cpu')
            model.set_env(env)
    
    print(f"✅ Modelo cargado: {algo_name}")
    print(f"   - Entrada: {model.policy.observation_space.shape}")
    print(f"   - Salida: {model.policy.action_space.shape}")
    
    return model, env

# ============ WRAPPER PARA ACCIONES DETERMINÍSTICAS ============
class DeterministicActionWrapper(gym.Wrapper):
    """
    Wrapper que fuerza acciones determinísticas (0 exploración) durante el entrenamiento.
    Intercepta las acciones en step() y las reemplaza por acciones determinísticas del modelo.
    """
    def __init__(self, env: gym.Env, model):
        super(DeterministicActionWrapper, self).__init__(env)
        self.model = model
        self.current_obs = None
    
    def step(self, action):
        """
        Intercepta la acción y la reemplaza por una determinística basada en la observación actual.
        """
        # Si tenemos la observación actual, usar acción determinística
        if self.current_obs is not None:
            try:
                # Obtener acción determinística del modelo basada en la observación actual
                deterministic_action, _ = self.model.predict(self.current_obs, deterministic=True)
                # Usar la acción determinística en lugar de la original
                action = deterministic_action
            except Exception as e:
                # Si falla, usar la acción original
                pass
        
        # Ejecutar step con la acción determinística
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Guardar observación actual para el próximo step
        self.current_obs = obs.copy() if obs is not None else None
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset del entorno y guardar observación inicial."""
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs.copy() if obs is not None else None
        return obs, info

# ============ ENTRENAMIENTO ONLINE ============
def train_online(model, env: gym.Env, config: Dict[str, Any]):
    """Entrenamiento online en producción"""
    train_config = config.get('training', {})
    
    total_timesteps = train_config.get('total_timesteps', 10000)
    eval_freq = train_config.get('eval_freq', 1000)
    n_eval_episodes = train_config.get('n_eval_episodes', 2)
    save_freq = train_config.get('save_freq', 5000)
    enable_exploration = train_config.get('enable_exploration', True)  # Por defecto con exploración
    
    print(f"\n🚀 INICIANDO ENTRENAMIENTO ONLINE")
    print(f"   Timesteps totales: {total_timesteps}")
    print(f"   Frecuencia evaluación: {eval_freq}")
    print(f"   Guardado cada: {save_freq} pasos")
    print(f"   Exploración: {'✅ ACTIVADA' if enable_exploration else '❌ DESACTIVADA (modo determinístico)'}")
    print(f"   Hora inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Directorio para checkpoints
    checkpoint_dir = f"./checkpoints/online/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Callback para guardar checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix='online_model'
    )
    
    # Entorno de evaluación (clon del entorno principal)
    eval_env = create_production_env(config)
    eval_env = apply_wrappers(eval_env, config)
    
    # Si el entorno principal fue adaptado, aplicar el mismo wrapper al de evaluación
    model_dim = model.observation_space.shape[0]
    eval_env_dim = eval_env.observation_space.shape[0]
    env_dim = env.observation_space.shape[0]
    
    # Si el entorno principal tiene dimensiones diferentes al modelo, fue adaptado
    if env_dim == model_dim and eval_env_dim != model_dim:
        # El entorno principal fue adaptado, aplicar el mismo wrapper al de evaluación
        eval_env = ObservationAdapterWrapper(eval_env, model_dim)
        print(f"   🔄 Entorno de evaluación adaptado: {eval_env_dim} → {model_dim} dimensiones")
    
    # Callback de evaluación
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir + '/best/',
        log_path=checkpoint_dir + '/logs/',
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Verificar que el entorno esté asignado antes de entrenar
    if not hasattr(model, 'env') or model.env is None:
        print(f"   ⚠️  El modelo no tiene entorno asignado. Intentando asignar...")
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv
            vec_env = DummyVecEnv([lambda: env])
            model.set_env(vec_env)
            print(f"   ✅ Entorno asignado antes de entrenar")
        except Exception as e:
            error_msg = str(e)
            if "Action spaces do not match" in error_msg or "Observation spaces do not match" in error_msg:
                # Intentar forzar la asignación ignorando las diferencias de espacios
                print(f"   ⚠️  Espacios diferentes detectados, intentando asignar de todas formas...")
                try:
                    # Crear un VecEnv que ignore las verificaciones estrictas
                    vec_env = DummyVecEnv([lambda: env])
                    # Asignar directamente sin verificación
                    model.env = vec_env
                    print(f"   ✅ Entorno asignado forzadamente (ignorando diferencias de espacios)")
                except Exception as e2:
                    print(f"   ❌ ERROR: No se pudo asignar el entorno al modelo: {e2}")
                    print(f"   ❌ No se puede entrenar sin entorno asignado")
                    raise RuntimeError("El modelo no tiene entorno asignado y no se pudo asignar") from e2
            else:
                print(f"   ❌ ERROR: No se pudo asignar el entorno al modelo: {e}")
                print(f"   ❌ No se puede entrenar sin entorno asignado")
                raise RuntimeError("El modelo no tiene entorno asignado y no se pudo asignar") from e
    
    # Configurar exploración si está desactivada
    callbacks_list = [checkpoint_callback, eval_callback]
    
    if not enable_exploration:
        print(f"\n   🔒 Modo determinístico activado - 0 exploración durante ejecución")
        print(f"   ✅ Las acciones serán determinísticas (sin exploración)")
        print(f"   ✅ El modelo seguirá entrenándose con los datos recopilados")
        
        # Envolver el entorno para forzar acciones determinísticas
        print(f"   🔄 Aplicando wrapper de acciones determinísticas...")
        env = DeterministicActionWrapper(env, model)
        
        # Reasignar el entorno al modelo con el wrapper
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv
            vec_env = DummyVecEnv([lambda: env])
            model.set_env(vec_env)
            print(f"   ✅ Entorno con wrapper determinístico asignado")
        except Exception as e:
            print(f"   ⚠️  No se pudo reasignar entorno: {e}")
            print(f"   ⚠️  Continuando con entorno original")
        
        # También establecer ent_coef muy bajo como respaldo
        if hasattr(model, 'ent_coef'):
            original_ent_coef = model.ent_coef
            if isinstance(model.ent_coef, str) and model.ent_coef == "auto":
                model.ent_coef = 0.0001
            else:
                model.ent_coef = 0.0001
            print(f"   📊 ent_coef ajustado a {model.ent_coef} (original: {original_ent_coef})")
    
    # Entrenar
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks_list,
            reset_num_timesteps=False,  # Continuar desde el modelo cargado
            log_interval=4,
            tb_log_name="online_production"
        )
        
        print(f"\n✅ Entrenamiento online completado")
        print(f"   Checkpoints guardados en: {checkpoint_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Guardar modelo final
        final_model_path = os.path.join(checkpoint_dir, 'model_final.zip')
        model.save(final_model_path)
        print(f"   Modelo final guardado: {final_model_path}")
        
        # Cerrar entorno de evaluación
        eval_env.close()
    
    return model

# ============ PRUEBA EN PRODUCCIÓN ============
def run_production_test(model, env: gym.Env, config: Dict[str, Any]):
    """Ejecuta una prueba en producción sin aprendizaje"""
    test_config = config.get('testing', {})
    n_episodes = test_config.get('n_episodes', 1)
    max_steps = test_config.get('max_steps', 288)  # 24 horas = 288 pasos (12 pasos/hora, 5 min/paso)
    render = test_config.get('render', True)
    
    print(f"\n🧪 EJECUTANDO PRUEBA EN PRODUCCIÓN")
    print(f"   Episodios: {n_episodes}")
    print(f"   Pasos máximos por episodio: {max_steps}")
    
    results = []
    
    for episode in range(n_episodes):
        print(f"\n📊 Episodio de prueba #{episode + 1}")
        
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_steps < max_steps:
            # Predecir acción (el wrapper ya adaptó la observación)
            action_from_model, _states = model.predict(obs, deterministic=True)
            
            print(f"\n   📥 Acción del modelo (RAW, sin procesar):")
            print(f"      Valores originales: {action_from_model}")
            print(f"      Dimensiones: {len(action_from_model)}")
            
            # Obtener action_space del modelo
            model_action_space = getattr(model, 'action_space', None)
            if model_action_space is not None:
                print(f"\n   📊 Action space del MODELO (con el que fue entrenado):")
                print(f"      Dimensiones: {model_action_space.shape[0]}")
                print(f"      Low:  {model_action_space.low}")
                print(f"      High: {model_action_space.high}")
                print(f"      Rango por dimensión:")
                for i in range(model_action_space.shape[0]):
                    print(f"         [{i}] [{model_action_space.low[i]:.1f}, {model_action_space.high[i]:.1f}]")
                
                print(f"\n   📊 Action space del ENTORNO (lo que necesita ahora):")
                print(f"      Dimensiones: {env.action_space.shape[0]}")
                print(f"      Low:  {env.action_space.low}")
                print(f"      High: {env.action_space.high}")
                print(f"      Rango por dimensión:")
                for i in range(env.action_space.shape[0]):
                    print(f"         [{i}] [{env.action_space.low[i]:.1f}, {env.action_space.high[i]:.1f}]")
                
                # Verificar si realmente hay un problema
                dims_match = model_action_space.shape[0] == env.action_space.shape[0]
                ranges_match = (np.allclose(model_action_space.low, env.action_space.low) and 
                               np.allclose(model_action_space.high, env.action_space.high))
                
                if dims_match and ranges_match:
                    print(f"\n   ✅ Action spaces coinciden perfectamente:")
                    print(f"      - Mismas dimensiones: {model_action_space.shape[0]}")
                    print(f"      - Mismos rangos: ✓")
                elif dims_match:
                    print(f"\n   ⚠️  ADVERTENCIA:")
                    print(f"      Mismas dimensiones ({model_action_space.shape[0]}) pero rangos diferentes")
                    print(f"      Se ajustarán los rangos durante la desnormalización")
                else:
                    print(f"\n   ❌ PROBLEMA CRÍTICO:")
                    print(f"      El modelo fue entrenado con {model_action_space.shape[0]} dimensiones")
                    print(f"      pero el entorno de producción necesita {env.action_space.shape[0]} dimensiones")
                    print(f"      Son actuadores DIFERENTES - necesitas definir un mapeo correcto.")
                    print(f"\n   💡 SOLUCIÓN:")
                    print(f"      1. Usar el mismo action_space del modelo en el entorno de producción")
                    print(f"      2. O definir un mapeo explícito de {model_action_space.shape[0]} → {env.action_space.shape[0]}")
                    print(f"      3. O reentrenar el modelo con el action_space del entorno de producción")
            
            # PROBLEMA: NormalizeAction wrapper usa np.rint() que redondea a enteros
            # SOLUCIÓN: Guardar la acción normalizada en info para que el bridge la use
            # En lugar de dejar que el wrapper la desnormalice incorrectamente
            
            # Verificar si el entorno tiene NormalizeAction wrapper
            has_normalize_action = False
            unwrapped_env = env
            while hasattr(unwrapped_env, 'env'):
                if hasattr(unwrapped_env, 'real_space'):
                    has_normalize_action = True
                    real_space = unwrapped_env.real_space
                    break
                unwrapped_env = unwrapped_env.env
            
            if has_normalize_action:
                # El wrapper NormalizeAction tiene problemas con np.rint()
                # Guardamos la acción normalizada en un atributo especial para que
                # el bridge la pueda usar directamente
                # Por ahora, pasamos la acción normalizada y el bridge intentará corregirla
                action = action_from_model
                print(f"   ✅ Pasando acción normalizada (wrapper la procesará, bridge intentará corregir)")
            else:
                # No hay wrapper, desnormalizar manualmente
                action = denormalize_action(
                    action_from_model, 
                    env.action_space, 
                    model_action_space=model_action_space,
                    verbose=True
                )
            
            # Guardar acción normalizada antes de pasarla al wrapper
            # para que el bridge pueda usarla directamente (evitando np.rint)
            if has_normalize_action:
                # Guardar en el entorno sin wrapper para que el bridge lo pueda acceder
                try:
                    unwrapped_env = env
                    while hasattr(unwrapped_env, 'env'):
                        unwrapped_env = unwrapped_env.env
                    # Guardar la acción normalizada en el entorno base
                    unwrapped_env._last_normalized_action = action_from_model.copy()
                except Exception as e:
                    print(f"   ⚠️  No se pudo guardar acción normalizada: {e}")
            
            # Ejecutar paso (el wrapper procesará la acción)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Guardar acción normalizada en info para debugging
            if has_normalize_action:
                info['action_normalized_original'] = action_from_model.tolist()
            
            episode_reward += reward
            episode_steps += 1
            
            if render:
                env.render()
            
            # Log cada 10 pasos
            if episode_steps % 10 == 0:
                print(f"   Paso {episode_steps}: Recompensa acumulada = {episode_reward:.2f}")
        
        # Guardar resultados del episodio
        episode_result = {
            'episode': episode + 1,
            'steps': episode_steps,
            'total_reward': float(episode_reward),
            'terminated': terminated,
            'truncated': truncated
        }
        results.append(episode_result)
        
        print(f"   ✅ Episodio completado:")
        print(f"      - Pasos: {episode_steps}")
        print(f"      - Recompensa total: {episode_reward:.2f}")
    
    # Guardar resultados
    results_file = f"./logs/production_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Resultados guardados en: {results_file}")
    return results

# ============ FUNCIÓN PRINCIPAL ============
def run_online_learning(config_path: str, model_path: Optional[str] = None):
    """Función principal de aprendizaje online"""
    print("\n" + "="*70)
    print("🚀 SISTEMA DE APRENDIZAJE ONLINE EN PRODUCCIÓN")
    print("="*70)
    
    # 1. Cargar configuración
    config = load_config(config_path)
    print(f"📋 Configuración cargada: {config.get('name', 'sin_nombre')}")
    ha_cfg = config.get('env_config', {}).get('production_config', {}).get('api_config', {}).get('homeassistant', {})
    if ha_cfg.get('use_flag_sync') or ha_cfg.get('flag_entity'):
        print(f"   🚩 Sincronización por bandera activa: {ha_cfg.get('flag_entity', 'N/A')} (esperar ON → leer sensores → modelo → actuadores → OFF)")
    
    # 2. Crear entorno
    env = create_production_env(config)
    
    # 3. Aplicar wrappers
    env = apply_wrappers(env, config)
    
    # 4. Cargar o crear modelo
    if model_path and os.path.exists(model_path):
        # Cargar modelo primero para obtener su action_space y usarlo en el entorno
        print(f"\n🔍 Verificando action_space del modelo para ajustar entorno...")
        try:
            temp_model = SAC.load(model_path, device='cpu')
            model_action_space = temp_model.action_space
            print(f"   📊 Modelo tiene action_space: {model_action_space.shape[0]} dimensiones")
            print(f"      Low:  {model_action_space.low}")
            print(f"      High: {model_action_space.high}")
            
            # Si el entorno tiene un action_space diferente, actualizarlo al del modelo
            if env.action_space.shape[0] != model_action_space.shape[0]:
                print(f"\n   ⚠️  El entorno tiene {env.action_space.shape[0]} dimensiones")
                print(f"   ⚠️  pero el modelo necesita {model_action_space.shape[0]} dimensiones")
                print(f"   🔄 Actualizando configuración para usar action_space del modelo...")
                
                # Actualizar la configuración para usar el action_space del modelo
                if 'env_config' not in config:
                    config['env_config'] = {}
                config['env_config']['action_space'] = {
                    'low': model_action_space.low.tolist(),
                    'high': model_action_space.high.tolist()
                }
                print(f"   ✅ Configuración actualizada")
                
                # Recrear el entorno con el action_space correcto
                env.close()
                env = create_production_env(config)
                env = apply_wrappers(env, config)
                print(f"   ✅ Entorno recreado con action_space del modelo ({model_action_space.shape[0]} dimensiones)")
        except Exception as e:
            print(f"   ⚠️  No se pudo verificar action_space del modelo: {e}")
            import traceback
            traceback.print_exc()
        
        # Ahora cargar el modelo normalmente
        model, env = load_production_model(model_path, config, env)
    else:
        print("\n⚠️  No se proporcionó modelo, creando uno nuevo...")
        if HAS_SB3:
            # Crear nuevo modelo
            algo_name = config.get('algorithm', {}).get('name', 'SAC')
            
            if algo_name == 'SAC':
                model = SAC('MlpPolicy', env, verbose=1, 
                          **config.get('algorithm_params', {}))
            elif algo_name == 'PPO':
                model = PPO('MlpPolicy', env, verbose=1,
                          **config.get('algorithm_params', {}))
            elif algo_name == 'TD3':
                model = TD3('MlpPolicy', env, verbose=1,
                          **config.get('algorithm_params', {}))
            elif algo_name == 'DDPG':
                model = DDPG('MlpPolicy', env, verbose=1,
                           **config.get('algorithm_params', {}))
            else:
                model = SAC('MlpPolicy', env, verbose=1)
            
            print(f"✅ Nuevo modelo creado: {algo_name}")
        else:
            print("❌ No se puede crear modelo sin Stable Baselines3")
            return
    
    # 5. Ejecutar prueba inicial (opcional)
    if config.get('run_initial_test', True):
        run_production_test(model, env, config)
    
    # 6. Entrenamiento online
    if config.get('enable_online_training', True):
        model = train_online(model, env, config)
    
    # 7. Ejecutar prueba final
    if config.get('run_final_test', True):
        run_production_test(model, env, config)
    
    # 8. Guardar modelo final
    final_save_path = f"./models/production_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
    model.save(final_save_path)
    print(f"\n💾 Modelo final guardado en: {final_save_path}")
    
    # 9. Cerrar entorno
    env.close()
    print("\n✅ Sistema de aprendizaje online finalizado")
    print(f"   Hora fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============ EJECUCIÓN DESDE LÍNEA DE COMANDOS ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de aprendizaje online en producción')
    parser.add_argument('--config', type=str, required=True, help='Ruta al archivo de configuración YAML')
    parser.add_argument('--model', type=str, help='Ruta al modelo preentrenado (.zip)')
    parser.add_argument('--test-only', action='store_true', help='Solo ejecutar prueba sin entrenar')
    parser.add_argument('--no-train', action='store_true', help='No realizar entrenamiento online')
    
    args = parser.parse_args()
    
    # Validar archivo de configuración
    if not os.path.exists(args.config):
        print(f"❌ Archivo de configuración no encontrado: {args.config}")
        sys.exit(1)
    
    # Validar modelo si se proporciona
    if args.model and not os.path.exists(args.model):
        print(f"⚠️  Modelo no encontrado: {args.model}")
        args.model = None
    
    # Modificar configuración según argumentos
    config = load_config(args.config)
    
    if args.test_only:
        config['enable_online_training'] = False
        config['run_initial_test'] = True
        config['run_final_test'] = True
    elif args.no_train:
        config['enable_online_training'] = False
    
    # Ejecutar
    try:
        run_online_learning(args.config, args.model)
    except KeyboardInterrupt:
        print("\n\n⚠️  Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)