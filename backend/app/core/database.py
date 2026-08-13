from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

# Database URL can be overridden via environment variable.
# Default uses an in-memory SQLite database for local testing.
DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


class Database:
    """Async database manager using SQLAlchemy."""

    def __init__(self, database_url: str = DEFAULT_DATABASE_URL):
        self.engine = create_async_engine(database_url, echo=False, future=True)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def create_tables(self) -> None:
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def get_session(self) -> AsyncSession:
        """Return a new async session."""
        return self.async_session()
