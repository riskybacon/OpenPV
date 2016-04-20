#ifndef BOILERPLATE_H_
#define BOILERPLATE_H_

#define PV_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName& source) = delete; \
    TypeName& operator=(const TypeName& source) = delete

#endif // BOILERPLATE_H_