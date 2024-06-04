# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file has contains dataset info and statistics for normalization."""

from collections.abc import Sequence


def get_task_name(domain_name: str) -> str:
  """Gets the task name for given domain."""
  if domain_name not in DMC_INFO:
    raise ValueError(f'Domain {domain_name} not in dataset.')
  return DMC_INFO[domain_name]['task_name']


def get_action_dim(domain_name: str) -> int:
  """Gets the action dimension for given domain."""
  if domain_name not in DMC_INFO:
    raise ValueError(f'Domain {domain_name} not in dataset.')
  return DMC_INFO[domain_name]['action_dim']


def get_state_dim(domain_name: str) -> int:
  """Gets the action dimension for given domain."""
  if domain_name not in DMC_INFO:
    raise ValueError(f'Domain {domain_name} not in dataset.')
  return DMC_INFO[domain_name]['state_dim']


def get_state_dim_no_velocity(domain_name: str) -> int:
  """Gets the action dimension for given domain."""
  if domain_name not in DMC_INFO:
    raise ValueError(f'Domain {domain_name} not in dataset.')
  return DMC_INFO[domain_name]['state_dim_no_velocity']


def get_state_fields(domain_name: str) -> Sequence[str]:
  """Gets the state fields for given domain."""
  if domain_name not in DMC_INFO:
    raise ValueError(f'Domain {domain_name} not in dataset.')
  return list(DMC_INFO[domain_name]['state_fields'].keys())


def get_state_fields_dims(domain_name: str) -> dict[str, int]:
  """Gets the state fields and each dimensions for given domain."""
  if domain_name not in DMC_INFO:
    raise ValueError(f'Domain {domain_name} not in dataset.')
  return DMC_INFO[domain_name]['state_fields']


def get_zero_dim_state_fields(domain_name: str) -> Sequence[str]:
  """Gets the zero-dimensional state fields for given domain."""
  if domain_name not in DMC_INFO:
    raise ValueError(f'Domain {domain_name} not in dataset.')
  return DMC_INFO[domain_name]['zero_dim_state_fields']


def get_state_mean(domain_name: str) -> Sequence[float]:
  """Gets the mean of the state over the dataset for given domain."""
  if domain_name not in DMC_VB_STATE_STATS:
    raise ValueError(f'Domain {domain_name} not in dataset.')
  return DMC_VB_STATE_STATS[domain_name]['mean']


def get_state_std(domain_name: str) -> Sequence[float]:
  """Gets the std of the state over the dataset for given domain."""
  if domain_name not in DMC_VB_STATE_STATS:
    raise ValueError(f'Domain {domain_name} not in dataset.')
  return DMC_VB_STATE_STATS[domain_name]['stddev']


DMC_INFO = {
    'walker': {
        'task_name': 'walk',
        'action_dim': 6,
        'state_dim': 24,
        'state_dim_no_velocity': 15,
        'state_fields': {
            'orientations': 14,
            'height': 1,
            'velocity': 9,
        },
        'zero_dim_state_fields': ['height'],
    },
    'cheetah': {
        'task_name': 'run',
        'action_dim': 6,
        'state_dim': 17,
        'state_dim_no_velocity': 8,
        'state_fields': {
            'position': 8,
            'velocity': 9,
        },
        'zero_dim_state_fields': [],
    },
    'humanoid': {
        'task_name': 'walk',
        'action_dim': 21,
        'state_dim': 67,
        'state_dim_no_velocity': 37,
        'state_fields': {
            'com_velocity': 3,
            'extremities': 12,
            'head_height': 1,
            'joint_angles': 21,
            'torso_vertical': 3,
            'velocity': 27,
        },
        'zero_dim_state_fields': ['head_height'],
    },
}


DMC_VB_STATE_STATS = {
    'cheetah': {
        'mean': [
            -0.0642446175357618,
            -0.0201884187354458,
            -0.044130660175722,
            -0.0834250080568031,
            -0.106605761133982,
            -0.17875323051813,
            0.0914475086285762,
            -0.058862532747006,
            4.51781252079175,
            0.00151153228541493,
            -0.0119096241923705,
            0.0123413184211005,
            -0.000124010280220808,
            -0.0157166972207928,
            -0.0202309778229998,
            0.00824039291209248,
            -0.00877146195889471,
        ],
        'stddev': [
            0.0524293678462371,
            0.162269243578419,
            0.325695799720639,
            0.309786391989642,
            0.294301146177371,
            0.195926931560251,
            0.274754503367339,
            0.286282860858787,
            4.11866924031,
            0.499899661706036,
            1.43194690310314,
            7.3708087352127,
            7.84756480939885,
            7.0168663706865,
            4.40911134620401,
            5.49458794028416,
            5.44983646292554,
        ],
    },
    'walker': {
        'mean': [
            0.564090377838074,
            0.0221260233439371,
            0.251421950370093,
            -0.294521763764138,
            0.342393574829508,
            0.325068005915253,
            0.304004776904524,
            0.20831875246902,
            0.511510411468729,
            0.206788579091816,
            0.267462559256487,
            0.387904063942428,
            0.345633758524617,
            0.102468321495031,
            0.771221359364604,
            -0.022705484538735,
            1.00246428529659,
            0.00548692133961244,
            0.00803389688721515,
            0.0895646889814122,
            0.0712516503451431,
            -0.00103848163048902,
            0.0158192794099301,
            0.0129084857628493,
        ],
        'stddev': [
            0.557691326540673,
            0.608640692244191,
            0.600898452508731,
            0.699265344001324,
            0.647605912272333,
            0.598094038165986,
            0.695059470459556,
            0.617314669553003,
            0.623481939353186,
            0.554000526251821,
            0.659744091111533,
            0.585431763616905,
            0.743148081208276,
            0.563764404979327,
            0.454379554526889,
            1.01735853714784,
            1.16677567476262,
            4.1876301580727,
            9.55205352481855,
            9.29955912161652,
            12.7807902669995,
            7.00451137198581,
            7.11066135508877,
            9.83981556891237,
        ],
    },
    'humanoid': {
        'mean': [
            -0.019172559178518,
            0.0250174078925482,
            -0.0278863604171865,
            0.104646050347222,
            0.177480710454092,
            0.091871261383483,
            -0.0646555937603959,
            0.32284528339155,
            -0.789509564204924,
            0.101723980424568,
            -0.187236407393546,
            0.0635423196575599,
            -0.100243931408017,
            -0.209899659015303,
            -0.967624646540253,
            0.882825619594145,
            0.248331173590319,
            0.0231849270209581,
            0.10358297792436,
            -0.191149131424651,
            0.323579006050399,
            0.0669397714986693,
            -0.586196867982785,
            -0.194039888972056,
            0.221689221816783,
            -0.156035629262309,
            -0.686396150407518,
            -0.50613189246507,
            -0.463329892298736,
            -0.134840176417997,
            0.239849087632028,
            0.243171331815536,
            -0.0034163495924817,
            0.375554500374252,
            -0.295385406270792,
            -0.381953379698935,
            0.402267641799734,
            0.082820494427811,
            0.0728904697116184,
            0.575729806012974,
            -0.0295503904300773,
            0.0286344506624771,
            -0.0346237304037758,
            0.0505890465163423,
            -0.180588025771374,
            -0.609941863668496,
            -0.131548085713339,
            0.10142024934506,
            -0.0679306846918143,
            -0.0634301913750624,
            -0.0566653798522746,
            -0.124656241553352,
            0.0147256096530897,
            0.0461092710412508,
            0.080416243684506,
            0.0234563594165835,
            0.00648455908008202,
            -0.102523189298486,
            0.0601708735442526,
            0.187009441728522,
            -0.239986556184506,
            -0.112053482877994,
            0.0575954614209082,
            -0.0227169274731786,
            -0.0599286628565785,
            -0.0868492169307219,
            -0.131023840340153,
        ],
        'stddev': [
            0.777357950416673,
            0.765906123255322,
            0.86746137608627,
            0.153218786039971,
            0.142645123343563,
            0.160897863573086,
            0.300052389359471,
            0.330013728167111,
            0.385558090835685,
            0.147263630450521,
            0.149665946570382,
            0.17545614879473,
            0.264713462787975,
            0.279241905684426,
            0.279619740176679,
            0.603830046984972,
            0.422890769170246,
            0.35975804019722,
            0.339633908631396,
            0.226472496104338,
            0.461503311770608,
            0.490237513553006,
            0.983567828009259,
            0.744441638181734,
            0.799883189165046,
            0.240672461294078,
            0.535611662827436,
            0.800548325546216,
            0.926028574019221,
            0.756740552006504,
            0.787249578402566,
            0.938635075351685,
            0.923891997583867,
            0.847564589034158,
            0.896632574361215,
            0.816072374177016,
            0.831545713397244,
            0.59714031264213,
            0.320484720243806,
            0.443936290602573,
            0.890706457974451,
            0.87822891559681,
            1.18342889876718,
            2.02840762788439,
            2.73102634512175,
            4.39230620788323,
            4.51977897918023,
            3.92550211995459,
            3.71499656267227,
            3.3198177686949,
            3.2646284140922,
            4.98319592996287,
            9.91664642424341,
            14.7556863995123,
            17.4967148792112,
            3.38271673173506,
            2.93770084882268,
            6.51247183447711,
            7.39150325267062,
            17.7939017169839,
            18.1468121814198,
            11.0323863152715,
            8.94461409001217,
            11.9016698863474,
            9.55968936303425,
            7.91488660277219,
            11.4673822372277,
        ],
    },
}
