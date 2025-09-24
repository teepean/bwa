#ifndef PORT_PTHREAD_H
#define PORT_PTHREAD_H

//--------------------------
#ifdef __cplusplus
extern "C"
{
#endif
//--------------------------

#ifdef _WIN32
// Windows pthread port implementation
#include <windows.h>

#define PTHREAD_CREATE_JOINABLE 0
#define PTHREAD_CANCELED ((void*)-1)

typedef struct __pthread_t
{
	HANDLE handle;
	DWORD thread_id;
	void *retval;
} pthread_t;

typedef struct __pthread_attr_t
{
	int detachstate;
} pthread_attr_t;

typedef struct __pthread_mutex_t
{
	CRITICAL_SECTION cs;
} pthread_mutex_t;

typedef struct __pthread_cond_t
{
	CONDITION_VARIABLE cv;
} pthread_cond_t;

int pthread_attr_init(pthread_attr_t * attr);
int pthread_attr_destroy(pthread_attr_t * attr);
int pthread_create(pthread_t * thread, const pthread_attr_t * attr, void *(*routine)(void*), void * arg);
int pthread_join(pthread_t thread, void ** retval);
int pthread_attr_setdetachstate(pthread_attr_t *attr, int detachstate);
void pthread_exit(void *retval);

int pthread_mutex_init(pthread_mutex_t *mutex, const void *attr);
int pthread_mutex_destroy(pthread_mutex_t *mutex);
int pthread_mutex_lock(pthread_mutex_t *mutex);
int pthread_mutex_unlock(pthread_mutex_t *mutex);

int pthread_cond_init(pthread_cond_t *cond, const void *attr);
int pthread_cond_destroy(pthread_cond_t *cond);
int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex);
int pthread_cond_broadcast(pthread_cond_t *cond);

#else
// Unix systems use standard pthread
#include <pthread.h>
#endif

//--------------------------
#ifdef __cplusplus
}
#endif
//--------------------------

#endif