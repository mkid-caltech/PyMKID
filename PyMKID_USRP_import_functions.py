from __future__ import division
import sys
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('/home/dmkidaq/GPU_SDR')
        import pyUSRP as u
    except ImportError:
        print "Cannot find the pyUSRP package"

def noise_run(rate,freq,front_end,tones,lapse,decimation,gain,vna,mode,pf,trigger,amplitudes,delay):

    if trigger is not None:
        try:
            trigger = eval('u.'+trigger+'()')
        except SyntaxError:
            u.print_error("Cannot find the trigger \'%s\'. Is it implemented in the USRP_triggers module?"%trigger)
            return ""
        except AttributeError:
            u.print_error("Cannot find the trigger \'%s\'. Is it implemented in the USRP_triggers module?"%trigger)
            return ""

    noise_filename = u.Get_noise(tones, measure_t = lapse, rate = rate, decimation = decimation, amplitudes = amplitudes,
                              RF = freq, output_filename = None, Front_end = front_end,Device = None, delay = delay*1e-9,
                              pf_average = pf, tx_gain = gain, mode = mode, trigger = trigger)
    if vna is not None:
        u.copy_resonator_group(vna, noise_filename)

    return noise_filename

def vna_run(gain,iter,rate,freq,front_end,f0,f1,lapse,points,ntones,delay_duration,delay_over='null'):

    if delay_over != 'null':
        delay = 1e-6*delay_over
        u.set_line_delay(rate, delay_over)
    else:
        try:
            if u.LINE_DELAY[str(int(rate/1e6))]: pass
        except KeyError:
            print "Cannot find line delay. Measuring line delay before VNA:"

            filename = u.measure_line_delay(rate, freq, front_end, USRP_num=0, tx_gain=0, rx_gain=0, output_filename=None, compensate = True, duration = delay_duration)

            delay = u.analyze_line_delay(filename,False)

            u.write_delay_to_file(filename, delay)

            u.load_delay_from_file(filename)

    if ntones ==1:
        ntones = None

    vna_filename = u.Single_VNA(start_f = f0, last_f = f1, measure_t = lapse, n_points = points, tx_gain = gain, Rate=rate, decimation=True, RF=freq, Front_end=front_end,
               Device=None, output_filename=None, Multitone_compensation=ntones, Iterations=iter, verbose=False)

    return vna_filename, delay*1e9
