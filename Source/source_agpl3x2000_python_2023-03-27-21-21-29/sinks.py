import logging

from webhelpers import text

from adhocracy import config
from adhocracy.lib import mail, microblog
from adhocracy.model import meta, Notification

TWITTER_LENGTH = 140
TRUNCATE_EXT = '...'

log = logging.getLogger(__name__)


def log_sink(pipeline):
    for notification in pipeline:
        log.debug("Generated notification: %s" % notification)
        yield notification


def twitter_sink(pipeline):
    twitter_enabled = bool(config.get('adhocracy.twitter.username', ''))
    for notification in pipeline:
        user = notification.user
        if (twitter_enabled and user.twitter
           and notification.priority >= user.twitter.priority):
            notification.language_context()
            short_url = microblog.shorten_url(notification.link)
            remaining_length = TWITTER_LENGTH - \
                (1 + len(short_url) + len(TRUNCATE_EXT))
            tweet = text.truncate(notification.subject, remaining_length,
                                  TRUNCATE_EXT, False)
            tweet += ' ' + short_url

            log.debug("twitter DM to %s: %s" % (user.twitter.screen_name,
                                                tweet))
            api = microblog.create_default()
            api.PostDirectMessage(user.twitter.screen_name, tweet)
        else:
            yield notification


def mail_sink(pipeline):
    for notification in pipeline:
        if notification.user.is_email_activated() and \
                notification.priority >= notification.user.email_priority:
            notification.language_context()
            headers = {'X-Notification-Id': notification.get_id(),
                       'X-Notification-Priority': str(notification.priority)}

            log.debug("mail to %s: %s" % (notification.user.email,
                                          notification.subject))
            mail.to_user(notification.user,
                         notification.subject,
                         notification.body,
                         headers=headers)

        else:
            yield notification


def database_sink(pipeline):
    for notification in pipeline:
        if meta.Session.query(Notification)\
                .filter(Notification.event_id == notification.event.id)\
                .filter(Notification.user_id == notification.user.id).all():
            log.warn('Notification already present: %s' % notification)
        else:
            meta.Session.add(notification)

        yield notification
