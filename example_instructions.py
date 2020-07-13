### Contain a set of example instructions that can be used to
### replace the rand_instructions in wfsim.

import wfsim
import numpy as np

def single_electron_instructions(c):
    """
    Instruction that is meant to simulate a single electron event.
    This is met by setting the instruction amplitude to 1, which may
    result in having 0 electrons (0 amplitude) in the simulated results.
    """
    n = c['nevents'] = c['event_rate'] * c['chunk_size'] * c['nchunk']
    c['total_time'] = c['chunk_size'] * c['nchunk']

    instructions = np.zeros(n, dtype=wfsim.instruction_dtype)
    uniform_times = c['total_time'] * (np.arange(n) + 0.5) / n
    instructions['time'] = np.repeat(uniform_times, 1) * int(1e9)

    # Actually holds which chunk this event occurred in
    instructions['event_number'] = np.digitize(instructions['time'],
            1e9 * np.arange(c['nchunk']) * c['chunk_size']) - 1

    instructions['type'] = np.tile([2], n) # S1 or S2
    instructions['recoil'] = ['er' for i in range(n)] # Kind of recoil

    # Random positioning
    r = np.sqrt(np.random.uniform(0, c['tpc_radius']**2, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    instructions['x'] = np.repeat(r * np.cos(t), 1)
    instructions['y'] = np.repeat(r * np.sin(t), 1)
    instructions['z'] = -10 * np.ones(n) # Choosing a constant depth

    instructions['amp'] = np.ones(n) # Best attempt for single electron

    return instructions

def kr83m_instructions(c):
    """
    Instruction that is meant to simulate Kr83m events in the TPC.
    """
    import nestpy
    half_life = 156.94e-9 # Kr intermediate state half-life in ns
    decay_energies = [32.2, 9.4] # Decay energies in KeV

    n = c['nevents'] = c['event_rate'] * c['chunk_size'] * c['nchunk']
    c['total_time'] = c['chunk_size'] * c['nchunk']

    # Uses 4*n to get the two energies for S1 and S2
    instructions = np.zeros(4*n, dtype=wfsim.instruction_dtype)
    instructions['event_number'] = np.digitize(instructions['time'],
            1e9 * np.arange(c['nchunk']) * c['chunk_size']) - 1

    instructions['type'] = np.tile([1,2], 2*n)
    instructions['recoil'] = ['er' for i in range(4*n)]

    # Random positioning
    r = np.sqrt(np.random.uniform(0, c['tpc_radius']**2, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    instructions['x'] = np.repeat(r * np.cos(t), 4)
    instructions['y'] = np.repeat(r * np.sin(t), 4)
    instructions['z'] = -10 * np.ones(4*n)
    # Choosing shallow z positioning

    # For correct times need to include the 156.94 ns half life of the intermediate state
    uniform_times = c['total_time'] * (np.arange(n) + 0.5) / n
    delayed_times = uniform_times + np.random.exponential(half_life/np.log(2),
            len(uniform_times))
    instructions['time'] = np.repeat(list(zip(uniform_times, delayed_times)), 2) * 1e9

    # Defining XENON-like detector
    nc = nestpy.NESTcalc(nestpy.VDetector())
    A = 131.293
    Z = 54
    density = 2.862 # g/cm^3    # SR1 Value
    drift_field = 82 # V/cm     # SR1 Value
    interaction = nestpy.INTERACTION_TYPE(7) # gamma

    energy = np.tile(decay_energies, n)
    quanta = []
    for en in energy:
        y = nc.GetYields(interaction,
                en,
                density,
                drift_field,
                A,
                Z,
                (1,1))
        quanta.append(nc.GetQuanta(y, density).photons)
        quanta.append(nc.GetQuanta(y, density).electrons)

    instructions['amp'] = quanta

    return instructions

def wall_instructions(c):
    """
    Instructions that are meant to focus on events near the walls of the TPC.
    """
    n = c['nevents'] = c['event_rate'] * c['chunk_size'] * c['nchunk']
    c['total_time'] = c['chunk_size'] * c['nchunk']

    instructions = np.zeros(n, dtype=wfsim.instruction_dtype)
    uniform_times = c['total_time'] * (np.arange(n) + 0.5) / n
    instructions['time'] = np.repeat(uniform_times, 1) * int(1e9)
    instructions['event_number'] = np.digitize(instructions['time'],
            1e9 * np.arange(c['nchunk']) * c['chunk_size']) - 1
    instructions['type'] = np.tile([2], n)
    instructions['recoil'] = ['er' for i in range(n)]

    r = np.sqrt(np.random.uniform( (c['tpc_radius'] - 5)**2, c['tpc_radius']**2, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    instructions['x'] = np.repeat(r * np.cos(t), 1)
    instructions['y'] = np.repeat(r * np.sin(t), 1)
    instructions['z'] = -10 * np.ones(n) # Constant depth choice

    instructions['amp'] = np.arange(1, 1000, n)

    return instructions

def uniform_electrons_instructions(c):
    """
    Instructions that is similar to single_electron_instructions, but has a uniform distribution of
    electrons from 1 to 1000. Should have the same pitfall as single_electron_instructions where a
    1 electron could simulate a 0 amplitude/electron event.
    """
    n = c['nevents'] = c['event_rate'] * c['chunk_size'] * c['nchunk']
    c['total_time'] = c['chunk_size'] * c['nchunk']

    instructions = np.zeros(n, dtype=wfsim.instruction_dtype)
    uniform_times = c['total_time'] * (np.arange(n) + 0.5) / n
    instructions['time'] = np.repeat(uniform_times, 1) * int(1e9)

    # Actually holds which chunk this event occurred in
    instructions['event_number'] = np.digitize(instructions['time'],
            1e9 * np.arange(c['nchunk']) * c['chunk_size']) - 1

    instructions['type'] = np.tile([2], n) # S1 or S2
    instructions['recoil'] = ['er' for i in range(n)] # Kind of recoil

    # Random positioning
    r = np.sqrt(np.random.uniform(0, c['tpc_radius']**2, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    instructions['x'] = np.repeat(r * np.cos(t), 1)
    instructions['y'] = np.repeat(r * np.sin(t), 1)
    instructions['z'] = -10 * np.ones(n) # Choosing a constant depth

    instructions['amp'] = np.random.uniform(1, 1000, n) # Best attempt for single electron

    return instructions
