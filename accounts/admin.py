from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    list_display = ['username', 'email', 'first_name', 'last_name', 'organization', 'rate_limit_tier', 'is_staff']
    list_filter = ['rate_limit_tier', 'is_staff', 'is_superuser', 'is_active', 'date_joined']
    search_fields = ['username', 'first_name', 'last_name', 'email', 'organization']
    
    fieldsets = UserAdmin.fieldsets + (
        ('Additional Info', {
            'fields': ('organization', 'api_key', 'rate_limit_tier')
        }),
    )
    
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Additional Info', {
            'fields': ('organization', 'rate_limit_tier')
        }),
    )
