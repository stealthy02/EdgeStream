#include "edgestream/core/signal.h"

volatile std::sig_atomic_t g_stop_requested = 0;

static void sigint_handler(int /*sig*/)
{
    g_stop_requested = 1;
}

void install_sigint_handler()
{
    std::signal(SIGINT, sigint_handler);
}

bool is_stop_requested()
{
    return g_stop_requested != 0;
}
