# Generated by Django 3.2.3 on 2021-07-15 16:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('manager', '0012_auto_20210430_1449'),
    ]

    operations = [
        migrations.AddField(
            model_name='set',
            name='comment',
            field=models.CharField(blank=True, max_length=200, verbose_name='Comment'),
        ),
    ]
