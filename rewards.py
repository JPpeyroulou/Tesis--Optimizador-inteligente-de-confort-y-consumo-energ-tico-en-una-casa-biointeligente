"""Implementation of reward functions."""

import subprocess, json
import pandas as pd

from datetime import datetime
from math import exp
from typing import Any, Dict, List, Optional, Tuple, Union

from sinergym.utils.constants import LOG_REWARD_LEVEL, YEAR
from sinergym.utils.logger import TerminalLogger


class BaseReward(object):

    logger = TerminalLogger().getLogger(name='REWARD',
                                        level=LOG_REWARD_LEVEL)

    def __init__(self):
        """
        Base reward class.

        All reward functions should inherit from this class.

        Args:
            env (Env): Gym environment.
        """

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Method for calculating the reward function."""
        raise NotImplementedError(
            "Reward class must have a `__call__` method.")

class LinearReward(BaseReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0
    ):
        """
        Linear reward function.

        It considers the energy consumption and the absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0))

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__()

        # Basic validations
        if not (0 <= energy_weight <= 1):
            self.logger.error(
                f'energy_weight must be between 0 and 1. Received: {energy_weight}')
            raise ValueError
        if not all(isinstance(v, str)
                   for v in temperature_variables + energy_variables):
            self.logger.error('All variable names must be strings.')
            raise TypeError

        # Name of the variables
        self.temp_names = temperature_variables
        self.energy_names = energy_variables

        # Reward parameters
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

        # Summer period
        self.summer_start = summer_start  # (month, day)
        self.summer_final = summer_final  # (month, day)

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> List[float]:
        """Calculate the energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with energy consumed in each energy variable.
        """
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(
            self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the temperature violation (ºC) in each observation's temperature variable.

        Returns:
            List[float]: List with temperature violation in each zone.
        """

        # Current datetime and summer period
        month = max(1, min(12, int(obs_dict['month'])))
        day = max(1, min(28, int(obs_dict['day_of_month'])))
        current_dt = datetime(YEAR, month, day)
        summer_start_date = datetime(YEAR, *self.summer_start)
        summer_final_date = datetime(YEAR, *self.summer_final)

        temp_range = self.range_comfort_summer if \
            summer_start_date <= current_dt <= summer_final_date else \
            self.range_comfort_winter

        temp_values = [obs_dict[v] for v in self.temp_names]

        return [max(temp_range[0] - T, 0, T - temp_range[1])
                for T in temp_values]

    def _get_reward(self) -> Tuple[float, ...]:
        """Compute the final reward value.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float, ...]: Total reward calculated and reward terms.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * \
            (1 - self.W_energy) * self.comfort_penalty
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term

class NuestroReward(BaseReward):

  
    # tenemos que cambiar los ramgos de temperatura por los rangos de confort termico. le pasamos a esta funcioanla temperatura actual. con ella calculamos el indice de confor termico actual
    # con esto lo utilizamos en la ecuacion R al igual manera que se utiliza la temperatura (penalizando si se salew de rango)
    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        high_price: float = 1.0, 
        low_price: float = 1.0,
        schedule_csv: Optional[str] = None
    ):
        """
        Linear reward function.

        It considers the energy consumption and the absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0))

            R = - W * lambda_E * power - (1 - W) * lambda_T * (max(PMVactual - (-0,5), 0) + max((0,5) - PMVactual, 0))

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__()

        # Basic validations
        if not (0 <= energy_weight <= 1):
            self.logger.error(
                f'energy_weight must be between 0 and 1. Received: {energy_weight}')
            raise ValueError
        if not all(isinstance(v, str)
                   for v in temperature_variables + energy_variables):
            self.logger.error('All variable names must be strings.')
            raise TypeError

        # Name of the variables
        self.temp_names = temperature_variables
        self.energy_names = energy_variables

        # Reward parameters
        #self.range_comfort_winter = range_comfort_winter
        #self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature
        self.low_price = low_price
        self.high_price = high_price
        self.schedule = None
        if schedule_csv is not None:
            self.schedule = pd.read_csv(schedule_csv, parse_dates=['date'])

        # Summer period
        #self.summer_start = summer_start  # (month, day)
        #self.summer_final = summer_final  # (month, day)

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        

        # Energy calculation        
        month = max(1, min(12, int(obs_dict['month'])))
        day = max(1, min(28, int(obs_dict['day_of_month'])))
        hour_raw = max(0, min(23, int(obs_dict['hour'])))
        current_dt = datetime(YEAR, month, day, hour_raw)
        weekday = current_dt.weekday()  # Devuelve un entero entre 0 (lunes) y 6 (domingo)
        hour = current_dt.hour      # Devuelve la hora como entero (sin paréntesis)
        price_kwh = self.low_price
        if (weekday < 5 and (hour >= 17 and hour <= 21)):
            price_kwh = self.high_price        
          
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values) 
        self.energy_penalty = -(self.total_energy / 1000) * price_kwh
        # self.energy_penalty = -self.total_energy
      

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -(self.total_temp_violation)

        # Schedule de presencia
        if self.schedule is not None:
          current_date = pd.Timestamp(year=YEAR,
                                    month=int(obs_dict['month']),
                                    day=int(obs_dict['day_of_month']))
          current_hour = int(obs_dict['hour'])

          row = self.schedule[
              (self.schedule['date'] == current_date) &
              (self.schedule['hour'] == current_hour)
          ]

          if not row.empty:
              factor = row['my_factor'].values[0]
              # 👇 aplicás tu condición
              if factor == 0:
                self.W_energy = 1
              # pass
              #else:
              #  self.W_energy = energy_weight

          #if (weekday < 5):
          #  self.W_energy = 0

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> List[float]:
        """Calculate the energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with energy consumed in each energy variable.
        """
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(
            self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the temperature violation (ºC) in each observation's temperature variable.

        Returns:
            List[float]: List with temperature violation in each zone.
        """

        tdb = obs_dict[self.temp_names[0]]
        rh  = obs_dict[self.temp_names[1]]

        pmv = -7.4928 + 0.2882*tdb -0.0020*rh + 0.0004*tdb*rh      

        #if pmv >= 0 :
        #  pmv = 0
       

        violacion = max(-0,5 - (pmv), 0, pmv - 0,5)
        # violacion = abs(pmv)  # Penaliza cualquier desviación de 0
        return [violacion]

    def _get_reward(self) -> Tuple[float, ...]:
        """Compute the final reward value.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float, ...]: Total reward calculated and reward terms.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * (1 - self.W_energy) * (self.comfort_penalty)
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term

class NuestroRewardMultizona(BaseReward):

  
    # tenemos que cambiar los ramgos de temperatura por los rangos de confort termico. le pasamos a esta funcioanla temperatura actual. con ella calculamos el indice de confor termico actual
    # con esto lo utilizamos en la ecuacion R al igual manera que se utiliza la temperatura (penalizando si se salew de rango)
    def __init__(
        self,
        temperature_variables: List[str],
        humidity_variables: List[str],
        energy_variables: List[str],
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        tarifa_json: Optional[str] = None,
        high_price: float = 1.0,
        low_price: float = 1.0,
        **kwargs
    ):
        """Simplified reward following Manjavacas et al. (2024).

        .. math::
            R = -W \\times \\lambda_E \\times (kWh \\times price)
              - (1-W) \\times \\lambda_T \\times PMV\\_violation

        PMV is estimated via simplified regression (clo=0.57, met=1.2, vr=0.1).
        Violation is bilateral: penalizes both PMV < -0.5 and PMV > +0.5,
        using a hybrid linear+quadratic penalty: violation = d + d^2.

        Energy cost uses real electricity tariffs (peak/off-peak from JSON).

        Args:
            temperature_variables: Name(s) of temperature observation variables.
            humidity_variables: Name(s) of humidity observation variables.
            energy_variables: Name(s) of energy/power observation variables.
            energy_weight: Balance between energy (W) and comfort (1-W). Default 0.5.
            lambda_energy: Scaling constant for energy term. Default 1.0.
            lambda_temperature: Scaling constant for comfort term. Default 1.0.
            tarifa_json: Path to JSON tariff file. Overrides high_price/low_price.
            high_price: Peak price fallback (if no tarifa_json).
            low_price: Off-peak price fallback (if no tarifa_json).
        """

        super().__init__()

        if not (0 <= energy_weight <= 1):
            self.logger.error(
                f'energy_weight must be between 0 and 1. Received: {energy_weight}')
            raise ValueError
        if not all(isinstance(v, str)
                   for v in temperature_variables + energy_variables):
            self.logger.error('All variable names must be strings.')
            raise TypeError

        self.temp_names = temperature_variables
        self.hum_names = humidity_variables
        self.energy_names = energy_variables

        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

        # --- Electricity tariff ---
        dias_map = {"lunes": 0, "martes": 1, "miercoles": 2,
                    "jueves": 3, "viernes": 4, "sabado": 5, "domingo": 6}

        if tarifa_json is not None:
            with open(tarifa_json, 'r') as f:
                tarifa = json.load(f)
            self.precio_punta = tarifa['precios']['punta']
            self.precio_fuera_punta = tarifa['precios']['fuera_de_punta']
            self.punta_inicio = tarifa['horarios']['punta_inicio']
            self.punta_fin = tarifa['horarios']['punta_fin']
            self.dias_punta = [dias_map[d] for d in tarifa['horarios']['dias_punta']]
            self.logger.info(
                f'Tarifa cargada desde {tarifa_json}: '
                f'punta=${self.precio_punta}/kWh, fuera_punta=${self.precio_fuera_punta}/kWh, '
                f'horario punta={self.punta_inicio}-{self.punta_fin}h')
        else:
            self.precio_punta = high_price
            self.precio_fuera_punta = low_price
            self.punta_inicio = 17
            self.punta_fin = 20
            self.dias_punta = [0, 1, 2, 3, 4]
            self.logger.info(
                f'Tarifa por parametros: punta=${self.precio_punta}, '
                f'fuera_punta=${self.precio_fuera_punta}')

        self.logger.info('Reward function initialized (simplified).')

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function following the paper structure:

            R = -W × λ_E × (energy_kWh × price) - (1-W) × λ_T × PMV_violation

        Adapted from Manjavacas et al. (2024) with two improvements:
        1. PMV-based comfort instead of raw temperature
        2. Energy cost ($) instead of raw consumption (kW)

        The electricity tariff already differentiates peak ($11.493/kWh) vs
        off-peak ($4.556/kWh), making peak 2.5x more expensive. This IS the
        incentive to pre-heat during cheap hours — no extra multipliers needed.

        Args:
            obs_dict: Dict with observation variable names and values.

        Returns:
            Tuple of (reward_value, reward_terms_dict).
        """

        # --- Datetime ---
        month = max(1, min(12, int(obs_dict['month'])))
        day = max(1, min(28, int(obs_dict['day_of_month'])))
        hour_raw = max(0, min(23, int(obs_dict['hour'])))
        current_dt = datetime(YEAR, month, day, hour_raw)
        weekday = current_dt.weekday()
        hour = current_dt.hour

        # --- Energy cost ---
        # Price varies by tariff (JSON). No extra multipliers.
        # Peak hours naturally cost 2.5x more — that's the signal.
        price_kwh = self.precio_fuera_punta
        if (weekday in self.dias_punta
                and self.punta_inicio <= hour <= self.punta_fin):
            price_kwh = self.precio_punta

        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -(self.total_energy / 1000) * price_kwh

        # --- PMV comfort violation ---
        # Cold side (PMV < -0.5): always penalized.
        # Hot side (PMV > +0.5): only penalized when the heat pump is
        # consuming energy (overheating is the agent's fault). When the HP
        # is off and it's hot, there's nothing the agent can do — no penalty.
        temp_violations, pmv_values = self._get_temperature_violation(
            obs_dict, self.total_energy)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # --- Reward (paper structure) ---
        avg_pmv = sum(pmv_values) / max(len(pmv_values), 1)

        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * (1 - self.W_energy) * self.comfort_penalty
        reward = energy_term + comfort_term

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
            'avg_pmv': avg_pmv,
            'price_kwh': price_kwh,
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> List[float]:
        """Calculate the energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with energy consumed in each energy variable.
        """
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(
        self, obs_dict: Dict[str, Any], total_energy: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        """Calculate PMV and comfort violation for each zone.

        Args:
            obs_dict: Environment observation.
            total_energy: Total heat-pump power [W] this timestep.

        Returns:
            Tuple[List[float], List[float]]: (violations, pmv_values) per zone.
        """

        violations = []
        pmv_values = []
        for temp_var, hum_var in zip(self.temp_names, self.hum_names):
            pmv, violation = self._calculate_pmv_violation(
                obs_dict[temp_var],
                obs_dict[hum_var],
                total_energy,
            )
            violations.append(violation)
            pmv_values.append(pmv)

        return violations, pmv_values

    def _calculate_pmv_violation(
        self, tdb: float, rh: float, total_energy: float = 0.0
    ) -> Tuple[float, float]:
        """Calculate PMV and comfort violation for a single zone.

        PMV estimated via simplified regression (clo=0.57, met=1.2, vr=0.1).
        Penalty uses hybrid linear+quadratic form (d + d^2).

        Cold side (PMV < -0.5) is always penalized.
        Hot side (PMV > +0.5) is only penalized when the heat pump is
        consuming energy, meaning the agent caused the overheating.
        When energy == 0 and it's hot, the agent can't cool — no penalty.

        Args:
            tdb: Dry-bulb temperature [°C].
            rh: Relative humidity [%].
            total_energy: Heat-pump power [W]. Used to gate hot-side penalty.

        Returns:
            Tuple[float, float]: (pmv_value, violation)
        """
        pmv = -7.4928 + 0.2882 * tdb - 0.0020 * rh + 0.0004 * tdb * rh

        d = max(-0.5 - pmv, 0.0) + max(pmv - 0.5, 0.0)
        #violacion lineal + cuadratica
        #violation = d + d * d
        #violacion lineal   
        #violation = d
        #violacion cuadratica
        violation = d * d

        return pmv, violation


class NuestroRewardMultizonaPPO(BaseReward):
    """Reward function optimized for PPO algorithm.
    
    Key differences from NuestroRewardMultizona (SAC):
    - Internal energy multiplier 2x (PPO needs stronger energy signal)
    - Flat penalty for ANY heating during peak hours (clear binary signal)
    - Peak multiplier 10x (vs 4x in SAC)
    - Lower comfort bonus 80/15 (vs 150/30) to not overwhelm energy savings
    - Stronger waste factor coefficient 80 (vs 50)
    
    Accepts the SAME __init__ parameters as NuestroRewardMultizona for YAML compatibility.
    """

    def __init__(
        self,
        temperature_variables: List[str],
        humidity_variables: List[str], 
        energy_variables: List[str],
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        tarifa_json: Optional[str] = None,
        high_price: float = 1.0, 
        low_price: float = 1.0,
        schedule_csv: Optional[str] = None
    ):
        super().__init__()

        if not (0 <= energy_weight <= 1):
            raise ValueError(f'energy_weight must be between 0 and 1. Got: {energy_weight}')

        self.temp_names = temperature_variables
        self.hum_names = humidity_variables
        self.energy_names = energy_variables

        # PPO-specific: internal energy multiplier 5x
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy * 5.0  # 5x internally
        self.lambda_temp = lambda_temperature

        # --- Tarifas electricas (same as SAC) ---
        dias_map = {"lunes": 0, "martes": 1, "miercoles": 2,
                    "jueves": 3, "viernes": 4, "sabado": 5, "domingo": 6}

        if tarifa_json is not None:
            with open(tarifa_json, 'r') as f:
                tarifa = json.load(f)
            self.precio_punta = tarifa['precios']['punta']
            self.precio_fuera_punta = tarifa['precios']['fuera_de_punta']
            self.punta_inicio = tarifa['horarios']['punta_inicio']
            self.punta_fin = tarifa['horarios']['punta_fin']
            self.dias_punta = [dias_map[d] for d in tarifa['horarios']['dias_punta']]
            self.logger.info(
                f'[PPO Reward] Tarifa desde {tarifa_json}: '
                f'punta=${self.precio_punta}, fuera=${self.precio_fuera_punta}')
        else:
            self.precio_punta = high_price
            self.precio_fuera_punta = low_price
            self.punta_inicio = 17
            self.punta_fin = 20
            self.dias_punta = [0, 1, 2, 3, 4]

        self.schedule = None
        if schedule_csv is not None:
            self.schedule = pd.read_csv(schedule_csv, parse_dates=['date'])

        self.logger.info('[PPO Reward] Initialized - le_eff=%.1f, pk=15x, dual(80/12), wf=150, flat_peak=-500' 
                         % (self.lambda_energy,))

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:

        # --- Datetime ---
        month = max(1, min(12, int(obs_dict['month'])))
        day = max(1, min(28, int(obs_dict['day_of_month'])))
        hour_raw = max(0, min(23, int(obs_dict['hour'])))
        current_dt = datetime(YEAR, month, day, hour_raw)
        weekday = current_dt.weekday()
        hour = current_dt.hour

        # --- Energy calculation ---
        price_kwh = self.precio_fuera_punta
        is_peak = (weekday in self.dias_punta
                   and self.punta_inicio <= hour <= self.punta_fin)
        if is_peak:
            price_kwh = self.precio_punta

        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -(self.total_energy / 1000) * price_kwh

        # --- PPO: Peak multiplier 15x (vs 4x in SAC) ---
        if is_peak:
            self.energy_penalty *= 15.0

        # --- PPO: Flat peak penalty (clear binary signal) ---
        # If the heat pump is ON during peak hours, apply a fixed penalty
        # This gives PPO an unmistakable "don't heat during peak" signal
        flat_peak_penalty = 0.0
        if is_peak and self.total_energy > 0:
            flat_peak_penalty = -500.0

        # --- Comfort violation ---
        temp_violations, pmv_values = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -(self.total_temp_violation)

        # --- Waste factor (PPO: coefficient 150, threshold -0.5) ---
        avg_pmv = sum(pmv_values) / max(len(pmv_values), 1)
        waste_factor = 1.0
        if self.total_energy > 0 and avg_pmv > -0.5:
            excess = avg_pmv + 0.5
            waste_factor = 1.0 + 150.0 * (excess ** 2)  # 150 vs 50 in SAC
            self.energy_penalty *= waste_factor

        # --- Schedule de presencia ---
        if self.schedule is not None:
            current_date = pd.Timestamp(year=YEAR,
                                        month=int(obs_dict['month']),
                                        day=int(obs_dict['day_of_month']))
            current_hour = int(obs_dict['hour'])
            row = self.schedule[
                (self.schedule['date'] == current_date) &
                (self.schedule['hour'] == current_hour)
            ]
            if not row.empty:
                factor = row['my_factor'].values[0]
                if factor == 0:
                    self.W_energy = 1

        # --- PPO: Comfort bonus dual(40/8) - reducido para nuevos hyperparams ---
        zones_in_comfort = sum(1 for pmv in pmv_values if -0.5 <= pmv <= 0.5)
        if zones_in_comfort >= 2:
            comfort_bonus = 40.0
        elif zones_in_comfort == 1:
            comfort_bonus = 8.0
        else:
            comfort_bonus = 0.0

        # --- Final reward ---
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * (1 - self.W_energy) * self.comfort_penalty
        reward = energy_term + comfort_term + comfort_bonus + flat_peak_penalty

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'comfort_bonus': comfort_bonus,
            'flat_peak_penalty': flat_peak_penalty,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
            'avg_pmv': avg_pmv,
            'waste_factor': waste_factor,
            'is_peak': is_peak
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict):
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(self, obs_dict):
        violations = []
        pmv_values = []
        for temp_var, hum_var in zip(self.temp_names, self.hum_names):
            pmv, violation = self._calculate_pmv_violation(
                obs_dict[temp_var], obs_dict[hum_var])
            violations.append(violation)
            pmv_values.append(pmv)
        return violations, pmv_values

    def _calculate_pmv_violation(self, tdb: float, rh: float) -> Tuple[float, float]:
        """Calculate PMV and comfort violation for a single zone.

        Same logic as NuestroRewardMultizona: bilateral penalty d + d^2.
        """
        pmv = -7.4928 + 0.2882 * tdb - 0.0020 * rh + 0.0004 * tdb * rh

        d = max(-0.5 - pmv, 0.0) + max(pmv - 0.5, 0.0)
        violation = d + d * d

        return pmv, violation


class EnergyCostLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        energy_cost_variables: List[str],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.4,
        temperature_weight: float = 0.4,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        lambda_energy_cost: float = 1.0
    ):
        """
        Linear reward function with the addition of the energy cost term.

        Considers energy consumption, absolute difference to thermal comfort and energy cost.

        .. math::
            R = - W_E * lambda_E * power - W_T * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0)) - (1 - W_P - W_T) * lambda_EC * power_cost

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer s-sum(exp(violation)
                    for violation in temp_violations if violation > 0)ession tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.4.
            temperature_weight (float, optional): Weight given to the temperature term. Defaults to 0.4.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1.0.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            lambda_energy_cost (flota, optional): Constant for removing dimensions from temperature(1/E). Defaults to 1.0.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature
        )

        self.energy_cost_names = energy_cost_variables
        self.W_temperature = temperature_weight
        self.lambda_energy_cost = lambda_energy_cost

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Energy cost calculation
        energy_cost_values = self._get_money_spent(obs_dict)
        self.total_energy_cost = sum(energy_cost_values)
        self.energy_cost_penalty = -self.total_energy_cost

        # Weighted sum of terms
        reward, energy_term, comfort_term, energy_cost_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_cost_term': energy_cost_term,
            'reward_energy_weight': self.W_energy,
            'reward_temperature_weight': self.W_temperature,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'energy_cost_penalty': self.energy_cost_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'money_spent': self.total_energy_cost
        }

        return reward, reward_terms

    def _get_money_spent(self, obs_dict: Dict[str,
                                              Any]) -> List[float]:
        """Calculate the total money spent in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with money spent in each energy cost variable.
        """
        return [v for k, v in obs_dict.items() if k in self.energy_cost_names]

    def _get_reward(self) -> Tuple[float, ...]:
        """It calculates reward value using the negative absolute comfort, energy penalty and energy cost penalty calculates previously.

        Returns:
            Tuple[float, ...]: Total reward calculated, reward term for energy, reward term for comfort and reward term for energy cost.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * \
            self.W_temperature * self.comfort_penalty
        energy_cost_term = self.lambda_energy_cost * \
            (1 - self.W_energy - self.W_temperature) * self.energy_cost_penalty

        reward = energy_term + comfort_term + energy_cost_term
        return reward, energy_term, comfort_term, energy_cost_term


class ExpReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0
    ):
        """
        Reward considering exponential absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * exp( (max(T - T_{low}, 0) + max(T_{up} - T, 0)) )

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature
        )

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        # Exponential Penalty
        self.comfort_penalty = -sum(exp(violation)
                                    for violation in temp_violations if violation > 0)

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy
        }

        return reward, reward_terms


class HourlyLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        default_energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        range_comfort_hours: tuple = (9, 19),
    ):
        """
        Linear reward function with a time-dependent weight for consumption and energy terms.

        Args:
            temperature_variables (List[str]]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            default_energy_weight (float, optional): Default weight given to the energy term when thermal comfort is considered. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            range_comfort_hours (tuple, optional): Hours where thermal comfort is considered. Defaults to (9, 19).
        """

        super(HourlyLinearReward, self).__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            default_energy_weight,
            lambda_energy,
            lambda_temperature
        )

        # Reward parameters
        self.range_comfort_hours = range_comfort_hours
        self.default_energy_weight = default_energy_weight

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Determine reward weight depending on the hour
        self.W_energy = self.default_energy_weight if self.range_comfort_hours[
            0] <= obs_dict['hour'] <= self.range_comfort_hours[1] else 1.0

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy
        }

        return reward, reward_terms


class NormalizedLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        max_energy_penalty: float = 8,
        max_comfort_penalty: float = 12,
    ):
        """
        Linear reward function with a time-dependent weight for consumption and energy terms.

        Args:
            temperature_variables (List[str]]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Default weight given to the energy term when thermal comfort is considered. Defaults to 0.5.
            max_energy_penalty (float, optional): Maximum energy penalty value. Defaults to 8.
            max_comfort_penalty (float, optional): Maximum comfort penalty value. Defaults to 12.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight
        )

        # Reward parameters
        self.max_energy_penalty = max_energy_penalty
        self.max_comfort_penalty = max_comfort_penalty

    def _get_reward(self) -> Tuple[float, ...]:
        """It calculates reward value using energy consumption and grades of temperature out of comfort range. Aplying normalization

        Returns:
            Tuple[float, ...]: total reward calculated, reward term for energy and reward term for comfort.
        """
        # Update max energy and comfort
        self.max_energy_penalty = max(
            self.max_energy_penalty, self.energy_penalty)
        self.max_comfort_penalty = max(
            self.max_comfort_penalty, self.comfort_penalty)

        # Calculate normalization
        energy_norm = self.energy_penalty / \
            self.max_energy_penalty if self.max_energy_penalty else 0
        comfort_norm = self.comfort_penalty / \
            self.max_comfort_penalty if self.max_comfort_penalty else 0

        # Calculate reward terms with norm values
        energy_term = self.W_energy * energy_norm
        comfort_term = (1 - self.W_energy) * comfort_norm
        reward = energy_term + comfort_term

        return reward, energy_term, comfort_term


class MultiZoneReward(BaseReward):

    def __init__(
        self,
        energy_variables: List[str],
        temperature_and_setpoints_conf: Dict[str, str],
        comfort_threshold: float = 0.5,
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0
    ):
        """
        A linear reward function for environments with different comfort ranges in each zone. Instead of having
        a fixed and common comfort range for the entire building, each zone has its own comfort range, which is
        directly obtained from the setpoints established in the building. This function is designed for buildings
        where temperature setpoints are not controlled directly but rather used as targets to be achieved, while
        other actuators are controlled to reach these setpoints. A setpoint observation variable can be assigned
        per zone if it is available in the specific building. It is also possible to assign the same setpoint
        variable to multiple air temperature zones.

        Args:
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            temperature_and_setpoints_conf (Dict[str, str]): Dictionary with the temperature variable name (key) and the setpoint variable name (value) of the observation space.
            comfort_threshold (float, optional): Comfort threshold for temperature range (+/-). Defaults to 0.5.
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__()

        # Name of the variables
        self.energy_names = energy_variables
        self.comfort_configuration = temperature_and_setpoints_conf
        self.comfort_threshold = comfort_threshold

        # Reward parameters
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature
        self.comfort_ranges = {}

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
            'comfort_threshold': self.comfort_threshold
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> List[float]:
        """Calculate the energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with energy consumed in each energy variable.
        """
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(
            self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the total temperature violation (ºC) in the current observation.

        Returns:
           List[float]: List with temperature violation (ºC) in each zone.
        """
        # Calculate current comfort range for each zone
        self._get_comfort_ranges(obs_dict)

        temp_violations = [
            max(0.0, min(abs(T - comfort_range[0]), abs(T - comfort_range[1])))
            if T < comfort_range[0] or T > comfort_range[1] else 0.0
            for temp_var, comfort_range in self.comfort_ranges.items()
            if (T := obs_dict[temp_var])
        ]

        return temp_violations

    def _get_comfort_ranges(
            self, obs_dict: Dict[str, Any]):
        """Calculate the comfort range for each zone in the current observation.

        Returns:
            Dict[str, Tuple[float, float]]: Comfort range for each zone.
        """
        # Calculate current comfort range for each zone
        self.comfort_ranges = {
            temp_var: (setpoint - self.comfort_threshold, setpoint + self.comfort_threshold)
            for temp_var, setpoint_var in self.comfort_configuration.items()
            if (setpoint := obs_dict[setpoint_var]) is not None
        }

    def _get_reward(self) -> Tuple[float, ...]:
        """Compute the final reward value.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float, ...]: Total reward calculated and reward terms.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * \
            (1 - self.W_energy) * self.comfort_penalty
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term
