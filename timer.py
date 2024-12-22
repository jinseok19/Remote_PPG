from time import perf_counter


class Timer:
    time_stamps = []
    window_size = 100
    fps = 30

    rppg_timer_t = 0
    rrsp_timer_t = 0

    @classmethod
    def set_time_stamp(cls):
        cls.time_stamps.append(perf_counter())
        cls.time_stamps = cls.time_stamps[-cls.window_size:]
        cls.fps = 30 if len(cls.time_stamps) == 1 else (len(cls.time_stamps) - 1) / (cls.time_stamps[-1] - cls.time_stamps[0])

    @classmethod
    def get_fps(cls):
        return cls.fps

    @classmethod
    def check_sec_ppg(cls):
        curr_t = perf_counter()

        if cls.rppg_timer_t == 0:
            cls.rppg_timer_t = curr_t
            return True
        elif (curr_t - cls.rppg_timer_t) > 1:
            cls.rppg_timer_t = curr_t
            return True
        else:
            return False

    @classmethod
    def check_sec_rsp(cls):
        curr_t = perf_counter()

        if cls.rrsp_timer_t == 0:
            cls.rrsp_timer_t = curr_t
            return True
        elif (curr_t - cls.rrsp_timer_t) > 1:
            cls.rrsp_timer_t = curr_t
            return True
        else:
            return False
