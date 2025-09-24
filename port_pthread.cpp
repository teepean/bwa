#include "port_pthread.h"

#ifdef _WIN32
#include <stdlib.h>

struct thread_arg_t {
	void *(*routine)(void*);
	void *arg;
	pthread_t *thread;
};

static DWORD WINAPI thread_wrapper(LPVOID param) {
	struct thread_arg_t *ta = (struct thread_arg_t*)param;
	ta->thread->retval = ta->routine(ta->arg);
	free(ta);
	return 0;
}

int pthread_attr_init(pthread_attr_t * attr) {
	if (!attr) return -1;
	attr->detachstate = PTHREAD_CREATE_JOINABLE;
	return 0;
}

int pthread_attr_destroy(pthread_attr_t * attr) {
	return 0;
}

int pthread_attr_setdetachstate(pthread_attr_t *attr, int detachstate) {
	if (!attr) return -1;
	attr->detachstate = detachstate;
	return 0;
}

int pthread_create(pthread_t * thread, const pthread_attr_t * attr, void *(*routine)(void*), void * arg) {
	if (!thread || !routine) return -1;

	struct thread_arg_t *ta = (struct thread_arg_t*)malloc(sizeof(struct thread_arg_t));
	ta->routine = routine;
	ta->arg = arg;
	ta->thread = thread;

	thread->handle = CreateThread(NULL, 0, thread_wrapper, ta, 0, &thread->thread_id);
	thread->retval = NULL;

	return thread->handle ? 0 : -1;
}

int pthread_join(pthread_t thread, void ** retval) {
	if (thread.handle) {
		WaitForSingleObject(thread.handle, INFINITE);
		if (retval) *retval = thread.retval;
		CloseHandle(thread.handle);
	}
	return 0;
}

void pthread_exit(void *retval) {
	ExitThread(0);
}

int pthread_mutex_init(pthread_mutex_t *mutex, const void *attr) {
	if (!mutex) return -1;
	InitializeCriticalSection(&mutex->cs);
	return 0;
}

int pthread_mutex_destroy(pthread_mutex_t *mutex) {
	if (!mutex) return -1;
	DeleteCriticalSection(&mutex->cs);
	return 0;
}

int pthread_mutex_lock(pthread_mutex_t *mutex) {
	if (!mutex) return -1;
	EnterCriticalSection(&mutex->cs);
	return 0;
}

int pthread_mutex_unlock(pthread_mutex_t *mutex) {
	if (!mutex) return -1;
	LeaveCriticalSection(&mutex->cs);
	return 0;
}

int pthread_cond_init(pthread_cond_t *cond, const void *attr) {
	if (!cond) return -1;
	InitializeConditionVariable(&cond->cv);
	return 0;
}

int pthread_cond_destroy(pthread_cond_t *cond) {
	// Windows condition variables don't need explicit cleanup
	return 0;
}

int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
	if (!cond || !mutex) return -1;
	return SleepConditionVariableCS(&cond->cv, &mutex->cs, INFINITE) ? 0 : -1;
}

int pthread_cond_broadcast(pthread_cond_t *cond) {
	if (!cond) return -1;
	WakeAllConditionVariable(&cond->cv);
	return 0;
}

#endif