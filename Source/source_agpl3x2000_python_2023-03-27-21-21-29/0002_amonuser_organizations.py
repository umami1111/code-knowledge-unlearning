# Generated by Django 2.0.2 on 2018-02-19 20:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('organizations', '__first__'),
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='amonuser',
            name='organizations',
            field=models.ManyToManyField(to='organizations.Organization'),
        ),
    ]
