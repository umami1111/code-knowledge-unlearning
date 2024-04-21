# -*- coding: utf-8 -*-

# Generated by Django 1.9.12 on 2017-04-02 16:44
from django.db import migrations, models
import wger.core.models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0009_auto_20160303_2340'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='birthdate',
            field=models.DateField(
                null=True,
                validators=[wger.core.models.profile.birthdate_validator],
                verbose_name='Date of Birth'
            ),
        ),
        migrations.AlterField(
            model_name='license',
            name='full_name',
            field=models.CharField(
                help_text=
                'If a license has been localized, e.g. the Creative Commons licenses for the different countries, add them as separate entries here.',
                max_length=60,
                verbose_name='Full name'
            ),
        ),
    ]
